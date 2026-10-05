#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>

// Dear ImGui Headers
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

struct BodiesSoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
    float *radius;
};

constexpr float G = 1.0f;
constexpr float BASE_DELTA_T = 0.02f;
constexpr float SOFTENING_SQ = 1e-4f;

// Helper to allocate SoA on CUDA Unified Memory
void allocate_bodies(BodiesSoA& bodies, int count) {
    size_t bytes = count * sizeof(float);
    cudaMallocManaged(&bodies.x, bytes);
    cudaMallocManaged(&bodies.y, bytes);
    cudaMallocManaged(&bodies.z, bytes);
    cudaMallocManaged(&bodies.vx, bytes);
    cudaMallocManaged(&bodies.vy, bytes);
    cudaMallocManaged(&bodies.vz, bytes);
    cudaMallocManaged(&bodies.mass, bytes);
    cudaMallocManaged(&bodies.radius, bytes);
}

// Helper to free CUDA memory
void free_bodies(BodiesSoA& bodies) {
    cudaFree(bodies.x); cudaFree(bodies.y); cudaFree(bodies.z);
    cudaFree(bodies.vx); cudaFree(bodies.vy); cudaFree(bodies.vz);
    cudaFree(bodies.mass); cudaFree(bodies.radius);
}

// Helper to initialize single body data
void init_body(BodiesSoA& bodies, int i) {
    if (i == 0) {
        // Heavy central star
        bodies.x[i] = 0.0f; bodies.y[i] = 0.0f; bodies.z[i] = 0.0f;
        bodies.vx[i] = 0.0f; bodies.vy[i] = 0.0f; bodies.vz[i] = 0.0f;
        bodies.mass[i] = 300.0f;
        bodies.radius[i] = 2.0f;
    } else {
        float px = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 80.0f;
        float py = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 80.0f;
        float pz = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 15.0f;

        bodies.x[i] = px;
        bodies.y[i] = py;
        bodies.z[i] = pz;

        // Varied masses (0.2 to 15.0)
        bodies.mass[i] = 0.2f + (static_cast<float>(rand()) / RAND_MAX) * 14.8f;
        bodies.radius[i] = 0.2f + bodies.mass[i] * 0.1f;

        // Tangential orbital velocity calculation
        float dist = std::sqrt(px * px + py * py) + 0.1f;
        float speed = std::sqrt(G * 300.0f / dist); // Orbital estimate around central mass
        
        bodies.vx[i] = -py / dist * speed + (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.3f;
        bodies.vy[i] =  px / dist * speed + (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.3f;
        bodies.vz[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
}

// CUDA Kernel: All-pairs N-body physics calculation
__global__ void nbody_step_kernel(BodiesSoA bodies, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float pos_i_x = bodies.x[i];
    float pos_i_y = bodies.y[i];
    float pos_i_z = bodies.z[i];

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int j = 0; j < N; ++j) {
        float dx = bodies.x[j] - pos_i_x;
        float dy = bodies.y[j] - pos_i_y;
        float dz = bodies.z[j] - pos_i_z;

        float dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING_SQ;
        float inv_dist = rsqrtf(dist_sq);
        float inv_dist_cube = inv_dist * inv_dist * inv_dist;

        // Force exerted by body j on body i
        float force_scalar = G * bodies.mass[j] * inv_dist_cube;

        fx += force_scalar * dx;
        fy += force_scalar * dy;
        fz += force_scalar * dz;
    }

    float vx_new = bodies.vx[i] + fx * dt;
    float vy_new = bodies.vy[i] + fy * dt;
    float vz_new = bodies.vz[i] + fz * dt;

    bodies.vx[i] = vx_new;
    bodies.vy[i] = vy_new;
    bodies.vz[i] = vz_new;

    bodies.x[i] = pos_i_x + vx_new * dt;
    bodies.y[i] = pos_i_y + vy_new * dt;
    bodies.z[i] = pos_i_z + vz_new * dt;
}

int main() {
    int N = 128; // Initial body count
    BodiesSoA h_bodies;
    allocate_bodies(h_bodies, N);

    for (int i = 0; i < N; ++i) {
        init_body(h_bodies, i);
    }

    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(1280, 720, "CUDA N-Body Control Center", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Simulation state
    bool is_running = true;
    float speed_multiplier = 1.0f;
    int add_count = 10; // Number of objects to add dynamically

    // Camera Orbit State
    float cam_yaw = 0.0f;
    float cam_pitch = 20.0f;
    float cam_dist = 120.0f;
    double last_mouse_x = 0.0, last_mouse_y = 0.0;
    bool is_dragging = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // --- Mouse Camera Orbit Logic ---
        if (!io.WantCaptureMouse) {
            // Scroll to Zoom
            if (io.MouseWheel != 0.0f) {
                cam_dist -= io.MouseWheel * 5.0f;
                cam_dist = std::max(10.0f, std::min(cam_dist, 500.0f));
            }

            // Drag Left Mouse Button to Rotate Camera
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                double current_x, current_y;
                glfwGetCursorPos(window, &current_x, &current_y);

                if (is_dragging) {
                    float dx = static_cast<float>(current_x - last_mouse_x);
                    float dy = static_cast<float>(current_y - last_mouse_y);

                    cam_yaw += dx * 0.4f;
                    cam_pitch += dy * 0.4f;
                    cam_pitch = std::max(-89.0f, std::min(cam_pitch, 89.0f));
                }
                last_mouse_x = current_x;
                last_mouse_y = current_y;
                is_dragging = true;
            } else {
                is_dragging = false;
            }
        }

        // --- CUDA Physics Step ---
        if (is_running && N > 0) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
            float dt = BASE_DELTA_T * speed_multiplier;

            nbody_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(h_bodies, N, dt);
            cudaDeviceSynchronize();
        }

        // --- GUI Frame Start ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Controls Window (Top-Left Corner)
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(380, 210), ImGuiCond_FirstUseEver);
        ImGui::Begin("Simulation Controls");

        // START / STOP Button
        if (is_running) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("STOP", ImVec2(100, 38))) is_running = false;
            ImGui::PopStyleColor(2);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.75f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.9f, 0.3f, 1.0f));
            if (ImGui::Button("START", ImVec2(100, 38))) is_running = true;
            ImGui::PopStyleColor(2);
        }

        ImGui::SameLine();
        ImGui::PushItemWidth(140.0f);
        ImGui::InputFloat("Speed", &speed_multiplier, 0.1f, 0.5f, "%.2fx");
        if (speed_multiplier < 0.0f) speed_multiplier = 0.0f;
        ImGui::PopItemWidth();

        ImGui::Separator();
        ImGui::Text("Camera: Drag Left-Click to rotate | Scroll to zoom");
        ImGui::Text("Cam Distance: %.1f | Pitch: %.1f | Yaw: %.1f", cam_dist, cam_pitch, cam_yaw);

        ImGui::Separator();
        // Dynamically add X objects
        ImGui::PushItemWidth(100.0f);
        ImGui::InputInt("##AddCount", &add_count);
        if (add_count < 1) add_count = 1;
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
        if (ImGui::Button("Add Objects")) {
            int old_N = N;
            int new_N = N + add_count;

            BodiesSoA temp_bodies;
            allocate_bodies(temp_bodies, new_N);

            // Copy old bodies over
            cudaMemcpy(temp_bodies.x, h_bodies.x, old_N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(temp_bodies.y, h_bodies.y, old_N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(temp_bodies.z, h_bodies.z, old_N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(temp_bodies.vx, h_bodies.vx, old_N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(temp_bodies.vy, h_bodies.vy, old_N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(temp_bodies.vz, h_bodies.vz, old_N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(temp_bodies.mass, h_bodies.mass, old_N * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(temp_bodies.radius, h_bodies.radius, old_N * sizeof(float), cudaMemcpyDeviceToDevice);

            free_bodies(h_bodies);
            h_bodies = temp_bodies;
            N = new_N;

            // Initialize new bodies
            for (int i = old_N; i < N; ++i) {
                init_body(h_bodies, i);
            }
        }
        ImGui::PopStyleColor();

        ImGui::End();

        // 2. Objects Inspector Table Window (Bottom-Left Corner, non-overlapping)
        ImGui::SetNextWindowPos(ImVec2(10, 230), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(480, 470), ImGuiCond_FirstUseEver);
        ImGui::Begin("Object Inspector");
        
        ImGui::Text("Total Objects Active: %d", N);

        if (ImGui::BeginTable("BodiesTable", 8, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 400))) {
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 35.0f);
            ImGui::TableSetupColumn("Mass", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("PosX");
            ImGui::TableSetupColumn("PosY");
            ImGui::TableSetupColumn("PosZ");
            ImGui::TableSetupColumn("VelX");
            ImGui::TableSetupColumn("VelY");
            ImGui::TableSetupColumn("VelZ");
            ImGui::TableHeadersRow();

            for (int i = 0; i < N; ++i) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%d", i);
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", h_bodies.mass[i]);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", h_bodies.x[i]);
                ImGui::TableSetColumnIndex(3); ImGui::Text("%.1f", h_bodies.y[i]);
                ImGui::TableSetColumnIndex(4); ImGui::Text("%.1f", h_bodies.z[i]);
                ImGui::TableSetColumnIndex(5); ImGui::Text("%.2f", h_bodies.vx[i]);
                ImGui::TableSetColumnIndex(6); ImGui::Text("%.2f", h_bodies.vy[i]);
                ImGui::TableSetColumnIndex(7); ImGui::Text("%.2f", h_bodies.vz[i]);
            }
            ImGui::EndTable();
        }
        ImGui::End();

        // --- Render 3D Scene ---
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.04f, 0.04f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = (float)display_w / (float)(display_h ? display_h : 1);

        float fov = 60.0f * 3.14159265f / 180.0f;
        float near_z = 1.0f, far_z = 1000.0f;
        float f = 1.0f / std::tan(fov / 2.0f);
        GLfloat proj[16] = {
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far_z + near_z) / (near_z - far_z), -1,
            0, 0, (2.0f * far_z * near_z) / (near_z - far_z), 0
        };
        glLoadMatrixf(proj);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Camera View Transforms
        glTranslatef(0.0f, 0.0f, -cam_dist);
        glRotatef(cam_pitch, 1.0f, 0.0f, 0.0f);
        glRotatef(cam_yaw, 0.0f, 1.0f, 0.0f);

        // Draw Bodies with Variable Sizes and Dynamic Mass-Based Colors
        glEnable(GL_PROGRAM_POINT_SIZE);
        for (int i = 0; i < N; ++i) {
            float mass = h_bodies.mass[i];
            
            // Scaled Point Size
            float point_size = std::clamp(mass * 0.8f + 3.0f, 3.0f, 24.0f);
            glPointSize(point_size);

            glBegin(GL_POINTS);
            if (mass > 100.0f) {
                glColor3f(1.0f, 0.85f, 0.2f); // Gold star center
            } else if (mass > 8.0f) {
                glColor3f(1.0f, 0.4f, 0.2f);  // Orange/red medium body
            } else if (mass > 3.0f) {
                glColor3f(0.2f, 0.8f, 0.4f);  // Green body
            } else {
                glColor3f(0.3f, 0.6f, 1.0f);  // Blue light body
            }
            glVertex3f(h_bodies.x[i], h_bodies.y[i], h_bodies.z[i]);
            glEnd();
        }

        // Render ImGui Overlay
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    free_bodies(h_bodies);
    return 0;
}