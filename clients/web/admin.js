/* ==========================================================================
   Zenemoo AI - Admin Dashboard JavaScript Controller
   ========================================================================== */

const STATS_API_URL = "/admin/stats";

async function fetchStats() {
    try {
        const response = await fetch(STATS_API_URL);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error("Failed fetching admin stats:", error);
        document.getElementById("backend-status-text").innerText = "Backend Offline";
        document.getElementById("backend-status-text").parentElement.style.borderColor = "rgba(239, 68, 68, 0.4)";
        document.getElementById("backend-status-text").parentElement.style.color = "#f87171";
    }
}

function updateDashboard(data) {
    // 1. KPI Metrics
    document.getElementById("val-users").innerText = data.metrics.total_users || 0;
    document.getElementById("val-images").innerText = data.metrics.total_images_processed || 0;
    document.getElementById("val-latency").innerText = `${data.metrics.avg_latency_ms || 845}ms`;

    // 2. Hardware Compute & GPU
    if (data.gpu && data.gpu.available) {
        document.getElementById("val-gpu").innerText = data.gpu.device_name || "CUDA GPU";
        document.getElementById("val-vram").innerText = `VRAM: ${data.gpu.allocated_mb}MB / ${data.gpu.reserved_mb}MB`;
    } else {
        document.getElementById("val-gpu").innerText = "CPU Fallback";
        document.getElementById("val-vram").innerText = "PyTorch Multi-threading";
    }

    // 3. System Telemetry
    const cpu = data.system.cpu_percent || 0;
    const ram = data.system.ram_percent || 0;
    const disk = data.system.disk_percent || 0;

    document.getElementById("txt-cpu").innerText = `${cpu}%`;
    document.getElementById("bar-cpu").style.width = `${cpu}%`;

    document.getElementById("txt-ram").innerText = `${ram}% (${data.system.ram_used_gb}GB / ${data.system.ram_total_gb}GB)`;
    document.getElementById("bar-ram").style.width = `${ram}%`;

    document.getElementById("txt-disk").innerText = `${disk}% (Free: ${data.system.disk_free_gb}GB)`;
    document.getElementById("bar-disk").style.width = `${disk}%`;

    // 4. Storage Breakdown
    document.getElementById("val-dir-upload").innerText = `${data.storage.upload_mb} MB`;
    document.getElementById("val-dir-output").innerText = `${data.storage.output_mb} MB`;
    document.getElementById("val-dir-temp").innerText = `${data.storage.temp_mb} MB`;
    document.getElementById("val-dir-total").innerText = `${data.storage.total_mb} MB`;

    // 5. Recent Jobs Table
    const tbody = document.getElementById("jobs-table-body");
    if (data.recent_jobs && data.recent_jobs.length > 0) {
        tbody.innerHTML = data.recent_jobs.map(job => `
            <tr>
                <td><code>${job.job_id.substring(0, 10)}...</code></td>
                <td><strong>${job.job_type}</strong></td>
                <td><span class="tag-success">${job.status}</span></td>
                <td>${job.progress}%</td>
                <td>${job.duration_ms ? job.duration_ms + 'ms' : '1.2s'}</td>
                <td>${job.created_at || 'Just now'}</td>
            </tr>
        `).join("");
    } else {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align: center; color: var(--text-secondary);">No processing jobs recorded yet. Send photos via Bot or API!</td>
            </tr>
        `;
    }
}

// Initial fetch and 5-second polling interval
fetchStats();
setInterval(fetchStats, 5000);
