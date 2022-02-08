#include <cuda_runtime.h>
#include <stdio.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <deps/sutil/vec_math.h>
#include <deps/sutil/random.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <deps/stb/stb_image_write.h>

#include "vec_math_helper.h"
#include "camera.h"
#include "trace.h"
#include "cuda_check.h"

/// Divide N by S, round up result.
#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)))

/// Number of iterations of the Menger sponge.
#define ITER_COUNT 6

/// Number of ray bounces.
#define BOUNCE_COUNT 10

__forceinline__ __device__ float sd_box(float3 p, float3 b)
{
    float3 q = fabsf(p) - b;
    return length(fmaxf(q, make_float3(0.f))) +
           fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f);
}

/// Returns the distance to a unit-sized menger sponge and a color value based
/// on the number of iterations of the closest
/// https://iquilezles.org/www/articles/menger/menger.htm
__forceinline__ __device__ float2 map(float3 p)
{
    float d = sd_box(p, make_float3(1.f));
    float col = 1.f;

    float s = 1.f;
    for (int m = 0; m < ITER_COUNT; m++) {
        float3 a = mod(p * s, 2.f) - make_float3(1.f);
        s *= 3.f;
        float3 r = fabsf(make_float3(1.f) - 3.f * fabsf(a));

        float da = fmaxf(r.x, r.y);
        float db = fmaxf(r.y, r.z);
        float dc = fmaxf(r.z, r.x);
        float c = (fminf(da, fminf(db, dc)) - 1.f) / s;

        if (c > d) {
            d = c;
            // assign a color based on the iteration count
            col = (1.f + float(m)) / float(ITER_COUNT + 1);
        }
    }

    return make_float2(d, col);
}

struct hit {
    float3 hitpoint;
    /// Whether the SDF is hit.
    int hit;
    float3 normal;
    /// Single-valued color.
    float color;
};

__forceinline__ __device__ hit trace(float3 origin, float3 direction)
{
    // slight offset to prevent from self-intersection
#define TMIN .1f
#define TMAX 1000.f
    hit h;
    for (float t = TMIN; t < TMAX;) {
        float3 p = origin + t * direction;
        float2 d = map(p);
        if (d.x < .001f) {
            h.hit = true;
            h.hitpoint = p;
            // find normal using central differences
            // https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            const float eps = .0001f;
            h.normal = normalize(make_float3(
                    map(p + make_float3(eps, 0.f, 0.f)).x -
                    map(p - make_float3(eps, 0.f, 0.f)).x,
                    map(p + make_float3(0.f, eps, 0.f)).x -
                    map(p - make_float3(0.f, eps, 0.f)).x,
                    map(p + make_float3(0.f, 0.f, eps)).x -
                    map(p - make_float3(0.f, 0.f, eps)).x));
            h.color = d.y;
            return h;
        }
        t += d.x;
    }

    h.hit = false;
    return h;
#undef TMAX
#undef TMIN
}

__global__ void generate_pixel(
        uint size_x, uint size_y, uint sample_count, uchar *image,
        camera *camera)
{
    uint2 idx = make_uint2(
            blockDim.x * blockIdx.x + threadIdx.x,
            blockDim.y * blockIdx.y + threadIdx.y);

    if (idx.x >= size_x || idx.y >= size_y) return;
    const uint image_idx = idx.y * size_x + idx.x;

    float3 accumulated_color = make_float3(0.f);

    for (int j = 0; j < sample_count; j++) {
        // initialize random based on sample index and image index
        uint seed = tea<16>(image_idx, j);

        // generate a ray though the pixel, randomly offset within the pixel
        float2 jitter = make_float2(rnd(seed), rnd(seed));
        float2 res = make_float2(float(size_x), float(size_y));
        float2 d = ((make_float2(idx.x, idx.y) + jitter) / res) * 2.f - 1.f;
        float3 ray_origin = camera->origin;
        float3 ray_direction = normalize(
                d.x * camera->u + d.y * camera->v + camera->w);

        float3 throughput = make_float3(1.f);
        float3 radiance = make_float3(0.f);

        // keep bounding until the maximum number of bounces is hit,
        // or the ray does not intersect with the sdf
        for (int i = 0; i < BOUNCE_COUNT; i++) {
            hit h = trace(ray_origin, ray_direction);

            if (!h.hit) {
                // 'sky' color
                const float3 color = make_float3(.6, .8f, 1.f);
                radiance += throughput * color;
                break;
            }

            // find a diffuse color based on the single color value
            const float3 diff_color = make_float3(
                    (1.f - h.color) * .5f, (1.f - h.color) * .3f,
                    h.color * .6f);

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

        accumulated_color += radiance / float(sample_count);
    }

    // convert linear to sRGB
    const float inv_gamma = 1.f / 2.4f;
    float3 rgb = make_float3(
            powf(accumulated_color.x, inv_gamma),
            powf(accumulated_color.y, inv_gamma),
            powf(accumulated_color.z, inv_gamma));
    rgb = clamp(rgb, 0.f, 1.f);

    // write to image buffer
    image[3 * image_idx + 0] = rgb.x * 255.f;
    image[3 * image_idx + 1] = rgb.y * 255.f;
    image[3 * image_idx + 2] = rgb.z * 255.f;
}

void generate(
        uint size_x, uint size_y, uint sample_count, const char *filename)
{
    // initialize camera
    camera cam;
    cam.origin = make_float3(2.1f, 0.f, 0.f);
    float3 target = make_float3(0.f);
    const float3 up = make_float3(0.f, 1.f, 0.f);
    float aspect = float(size_x) / float(size_y);
    cam.w = normalize(target - cam.origin); // lookat direction
    cam.u = normalize(cross(cam.w, up)) * aspect; // screen right
    cam.v = normalize(cross(cam.u, cam.w)); // screen up

    // copy camera parameters to device
    camera *d_cam = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cam, sizeof(camera)));
    CUDA_CHECK(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));

    // create output buffer on device
    uchar *d_image = nullptr;
    size_t image_size = sizeof(char) * 3 * size_x * size_y;
    CUDA_CHECK(cudaMalloc(&d_image, image_size));
    CUDA_CHECK(cudaMemset(d_image, 0, image_size));

    // events for measuring elapsed time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // launch kernel
    dim3 block_size(16, 16, 1);
    dim3 block_count(ROUND_UP(size_x, 16), ROUND_UP(size_y, 16), 1);
    CUDA_CHECK(cudaEventRecord(start));
    generate_pixel<<<block_count, block_size>>>(
            size_x, size_y, sample_count, d_image, d_cam);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("kernel took %fs\n", milliseconds * 1e-3f);

    // when kernel is done, copy buffer back to host
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaFree(d_cam));
    uchar *image = (uchar *) malloc(image_size);
    CUDA_CHECK(cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_image));

    // write buffer to file
    stbi_flip_vertically_on_write(1);
    stbi_write_png(filename, size_x, size_y, 3, image, size_x * 3);
    free(image);
}

