#include <cuda_runtime.h>
#include <stdio.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <sutil/vec_math.h>
#include <sutil/random.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb/stb_image_write.h>

#include "vec_math_helper.h"
#include "camera.h"
#include "trace.h"
#include "cuda_check.h"

/// Divide N by S, round up result.
#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)))

/// SDF of a box.
__forceinline__ __device__ float sd_box(float3 p, float3 b)
{
    float3 q = fabsf(p) - b;
    return length(fmaxf(q, make_float3(0.f))) +
           fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f);
}

/// Returns the distance to a unit-sized menger sponge.
/// https://iquilezles.org/www/articles/menger/menger.htm
__forceinline__ __device__ float map(float3 p)
{
    // start with a unit box
    float d = sd_box(p, make_float3(1.f));

    float scale = 1.f;
    for (int m = 0; m < ITER_COUNT; m++) {
        // optimized, negative part of scaled sdf of three infinite boxes
        float3 a = mod(p * scale, 2.f) - make_float3(1.f);
        scale *= 3.f;
        float3 r = fabsf(make_float3(1.f) - 3.f * fabsf(a));

        float da = fmaxf(r.x, r.y);
        float db = fmaxf(r.y, r.z);
        float dc = fmaxf(r.z, r.x);
        float c = (fminf(da, fminf(db, dc)) - 1.f) / scale;

        d = fmaxf(d, c);
    }

    return d;
}

struct hit {
    float3 hitpoint;
    /// Whether the SDF is hit.
    int hit;
    float3 normal;
};

/// March a ray and return the hitpoint with the SDF.
__forceinline__ __device__ hit trace(float3 origin, float3 direction)
{
    // slight offset to prevent self-intersection
#define TMIN .01f
#define TMAX 1000.f
    hit h;
    for (float t = TMIN; t < TMAX;) {
        float3 p = origin + t * direction;
        float d = map(p);
        if (d < .001f) {
            h.hit = true;
            h.hitpoint = p;
            // find normal using central differences
            // https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            const float eps = .0001f;
            h.normal = normalize(make_float3(
                    map(p + make_float3(eps, 0.f, 0.f)) -
                    map(p - make_float3(eps, 0.f, 0.f)),
                    map(p + make_float3(0.f, eps, 0.f)) -
                    map(p - make_float3(0.f, eps, 0.f)),
                    map(p + make_float3(0.f, 0.f, eps)) -
                    map(p - make_float3(0.f, 0.f, eps))));
            return h;
        }
        // advance the ray with the distance to the sdf, since we know that we
        // won't skip intersections doing this
        t += d;
    }

    h.hit = false;
    return h;
#undef TMAX
#undef TMIN
}

/// Generates a radiance value for the ith sample of this pixel.
__forceinline__ __device__ float3 generate_pixel(
        uint image_idx, uint image_idx_x, uint image_idx_y, uint sample_idx,
        uint size_x, uint size_y, camera *camera)
{
    // initialize random based on sample index and image index
    uint seed = tea<16>(image_idx, sample_idx);

    // generate a ray though the pixel, randomly offset within the pixel
    float2 jitter = make_float2(rnd(seed), rnd(seed));
    float2 res = make_float2(size_x, size_y);
    float2 idx = make_float2(image_idx_x, image_idx_y);
    float2 d = ((idx + jitter) / res) * 2.f - 1.f; // position on raster
    float3 ray_origin = camera->origin;
    float3 ray_direction = normalize(
            d.x * camera->u + d.y * camera->v + camera->w);

    float3 throughput = make_float3(1.f);
    float3 radiance = make_float3(0.f);

    // keep bounding until the maximum number of bounces is hit,
    // or the ray does not intersect with the sdf
    for (int i = 0; i < MAX_BOUNCE_COUNT; i++) {
        hit h = trace(ray_origin, ray_direction);

        if (!h.hit) {
            // 'sky' color
            const float3 color = make_float3(.6, .8f, 1.f);
            radiance += throughput * color;
            break;
        }

        // pick a static diffuse color
        float3 diff_color = make_float3(.2f, .3f, .9f);

        // check if we continue using russian roulette, where the max component
        // of the color dictates the probability
        if (i > MIN_BOUNCE_COUNT) {
            float rr_prob = fmaxf(diff_color);
            if (rr_prob < rnd(seed)) break;
            // if continuing, scale with probability
            throughput /= rr_prob;
        }

        // surface model is lambertian, attenuation is equal to diffuse
        // color, assuming we sampled with cosine weighted hemisphere
        throughput *= diff_color;

        // set new origin and generate new direction
        ray_origin = h.hitpoint;
        float3 w_in = cosine_sample_hemisphere(rnd(seed), rnd(seed));
        frame onb(h.normal);
        onb.inverse_transform(w_in);
        ray_direction = w_in;
    }

    return radiance;
}

/// Implementation with regeneration: create a number of persistent threads that
/// complete samples one by one, starting new ones when the current one is
/// terminated.
__global__ void generate_pixel_regeneration(
        uint size_x, uint size_y, uint sample_count, float *buffer,
        camera *camera, unsigned long long int *idx)
{
    const ulong max_count = size_x * size_y * sample_count;
    while (true) {
        // obtain the next index. if is it out of bounds, stop
        unsigned long long int this_idx = atomicAdd(idx, 1);
        if (this_idx >= max_count) break;

        uint sample_idx = this_idx / (size_x * size_y);
        uint image_idx = this_idx - sample_idx * size_x * size_y;
        uint image_idx_y = image_idx / size_x;
        uint image_idx_x = image_idx - image_idx_y * size_x;

        // obtain radiance
        float3 radiance = generate_pixel(
                image_idx, image_idx_x, image_idx_y, sample_idx,
                size_x, size_y, camera);

        // atomically add to buffer
        atomicAdd(&buffer[4 * image_idx + 0], radiance.x / float(sample_count));
        atomicAdd(&buffer[4 * image_idx + 1], radiance.y / float(sample_count));
        atomicAdd(&buffer[4 * image_idx + 2], radiance.z / float(sample_count));
    }
}

/// Naive implementation: create a number of threads at least equal to the
/// number of pixels * the number of samples, each thread computes all samples
/// for that pixel.
__global__ void generate_pixel_naive(
        uint size_x, uint size_y, uint sample_count, float *buffer,
        camera *camera)
{
    uint2 idx = make_uint2(
            blockDim.x * blockIdx.x + threadIdx.x,
            blockDim.y * blockIdx.y + threadIdx.y);

    if (idx.x >= size_x || idx.y >= size_y) return;
    const uint image_idx = idx.y * size_x + idx.x;

    float3 accumulated_color = make_float3(0.f);

    for (int i = 0; i < sample_count; i++) {
        float3 radiance = generate_pixel(
                image_idx, idx.x, idx.y, i, size_x, size_y, camera);

        accumulated_color += radiance / float(sample_count);
    }

    // write to buffer
    buffer[4 * image_idx + 0] = accumulated_color.x;
    buffer[4 * image_idx + 1] = accumulated_color.y;
    buffer[4 * image_idx + 2] = accumulated_color.z;
}

/// Converts a linear radiance value to a sRGB pixel value.
uchar radiance_to_srgb(float val)
{ return (uchar) (clamp(powf(val, 1.f / 2.4f), 0.f, 1.f) * 255.f); }

void generate(
        uint size_x, uint size_y, uint sample_count, const char *filename)
{
    // variables that define the camera
    float3 origin = make_float3(-.3f);
    float3 target = make_float3(1.f);
    float3 up = make_float3(0.f, 1.f, 0.f);

    // initialize camera based on the above variables
    float aspect = float(size_x) / float(size_y);
    camera cam;
    cam.origin = origin;
    cam.w = normalize(target - cam.origin);       // lookat direction
    cam.u = normalize(cross(cam.w, up)) * aspect; // screen right
    cam.v = normalize(cross(cam.u, cam.w));       // screen up

    // copy camera parameters to device
    camera *d_cam = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cam, sizeof(camera)));
    CUDA_CHECK(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));

    // create output buffer on device
    float *d_buffer = nullptr;
    size_t buffer_size = sizeof(float) * 4 * size_x * size_y;
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
    CUDA_CHECK(cudaMemset(d_buffer, 0, buffer_size));

    // events for measuring elapsed time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // declare bool to get warnings even in unused code
#ifdef NAIVE
    bool do_naive = true;
#else
    bool do_naive = false;
#endif

    // launch kernel
    CUDA_CHECK(cudaEventRecord(start));
    unsigned long long int *d_idx = nullptr;
    if (do_naive) {
#define BLOCK_SIZE 16
        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 block_count(
                ROUND_UP(size_x, BLOCK_SIZE), ROUND_UP(size_y, BLOCK_SIZE), 1);
        generate_pixel_naive<<<block_count, block_size>>>(
                size_x, size_y, sample_count, d_buffer, d_cam);
#undef BLOCK_SIZE
    } else {
        // additionally, allocate a single long int counter
        CUDA_CHECK(cudaMalloc(&d_idx, sizeof(unsigned long long int)));
        CUDA_CHECK(cudaMemset(d_idx, 0, sizeof(unsigned long long int)));
        generate_pixel_regeneration<<<1024, 1024>>>(
                size_x, size_y, sample_count, d_buffer, d_cam, d_idx);
    }
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("kernel took %fs\n", milliseconds * 1e-3f);

    // when kernel is done, copy buffer back to host
    CUDA_SYNC_CHECK();
    if (!do_naive) CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_cam));
    float *buffer = (float *) malloc(buffer_size);
    CUDA_CHECK(cudaMemcpy(
            buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buffer));

    // convert buffer to format accepted by image writer
    uchar *image = (uchar *) malloc(sizeof(char) * 3 * size_x * size_y);
    for (uint i = 0; i < size_x * size_y; i++) {
        image[3 * i + 0] = radiance_to_srgb(buffer[4 * i + 0]);
        image[3 * i + 1] = radiance_to_srgb(buffer[4 * i + 1]);
        image[3 * i + 2] = radiance_to_srgb(buffer[4 * i + 2]);
    }
    free(buffer);

    // write buffer to file
    stbi_flip_vertically_on_write(1);
    stbi_write_png(filename, size_x, size_y, 3, image, size_x * 3);
    free(image);
}
