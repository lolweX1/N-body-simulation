#include <sycl/sycl.hpp>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdlib>

// Particle structure shared between CPU and SYCL
struct Particle {
    sycl::float4 pos; // x, y, z, mass
    sycl::float4 vel; // vx, vy, vz, unused
};

// SYCL Kernel to compute gravitational forces & update positions
void update_particles_sycl(sycl::queue& q, Particle* particles, int num_bodies, float dt, float softening) {
    q.parallel_for(sycl::range<1>(num_bodies), [=](sycl::id<1> idx) {
        int i = idx[0];
        sycl::float4 p_i = particles[i].pos;
        sycl::float3 acc = {0.0f, 0.0f, 0.0f};

        // Compute gravitational interaction with all other particles
        for (int j = 0; j < num_bodies; ++j) {
            sycl::float4 p_j = particles[j].pos;
            
            sycl::float3 r = {p_j.x() - p_i.x(), p_j.y() - p_i.y(), p_j.z() - p_i.z()};
            float dist_sq = r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + softening * softening;
            float inv_dist = 1.0f / std::sqrt(dist_sq);
            float inv_dist_cubed = inv_dist * inv_dist * inv_dist;

            float force = p_j.w() * inv_dist_cubed; // p_j.w is mass
            acc.x() += r.x() * force;
            acc.y() += r.y() * force;
            acc.z() += r.z() * force;
        }

        // Integration (Euler step)
        particles[i].vel.x() += acc.x() * dt;
        particles[i].vel.y() += acc.y() * dt;
        particles[i].vel.z() += acc.z() * dt;

        particles[i].pos.x() += particles[i].vel.x() * dt;
        particles[i].pos.y() += particles[i].vel.y() * dt;
        particles[i].pos.z() += particles[i].vel.z() * dt;
    });
}

int main() {
    // 1. Initialize GLFW & Window
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "N-Body Simulation (SYCL)", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // 2. Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // 3. Initialize SYCL Queue (Targets Intel Arc or default SYCL GPU)
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // 4. Setup Simulation Data
    const int NUM_BODIES = 1000;
    float dt = 0.005f;
    float softening = 0.1f;

    // Allocate Unified Shared Memory (USM) accessible by both CPU and SYCL GPU
    Particle* d_particles = sycl::malloc_shared<Particle>(NUM_BODIES, q);

    // Initialize random particle positions & velocities
    const float rand_max_f = static_cast<float>(RAND_MAX);
    for (int i = 0; i < NUM_BODIES; ++i) {
        float angle = (static_cast<float>(rand()) / rand_max_f) * 2.0f * 3.14159f;
        float dist = 0.5f + (static_cast<float>(rand()) / rand_max_f) * 2.0f;
        
        d_particles[i].pos = { dist * std::cos(angle), dist * std::sin(angle), 0.0f, 1.0f };
        d_particles[i].vel = { -std::sin(angle) * 0.5f, std::cos(angle) * 0.5f, 0.0f, 0.0f };
    }

    // 5. Main Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Run SYCL kernel calculation
        update_particles_sycl(q, d_particles, NUM_BODIES, dt, softening);
        q.wait(); // Synchronize kernel completion

        // Start ImGui Frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Control Panel UI
        ImGui::Begin("Simulation Controls");
        ImGui::Text("Bodies: %d", NUM_BODIES);
        ImGui::SliderFloat("Time Step (dt)", &dt, 0.001f, 0.02f);
        ImGui::SliderFloat("Softening", &softening, 0.01f, 0.5f);
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render particles as points
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        glColor3f(0.8f, 0.8f, 1.0f);
        for (int i = 0; i < NUM_BODIES; ++i) {
            glVertex3f(d_particles[i].pos.x(), d_particles[i].pos.y(), d_particles[i].pos.z());
        }
        glEnd();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    sycl::free(d_particles, q);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}