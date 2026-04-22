% ========================================================================
% 2D (x–y) ANGULAR SPECTRUM FOR k‑Wave-MATCHED GEOMETRY
% ------------------------------------------------------------------------
% - Linear acoustics, monochromatic (single frequency)
% - Exact point sources via k-space phasors (sub‑voxel)
% - Attenuation alpha_power = 1 (dB/(cm*MHz)) -> Nepers/m
% - Samples a true circular sensor arc as in your k‑Wave mask
% - Produces equivalent figure & CSV (column names & meaning)
% ========================================================================

clear; clc;

%% ----------------------- PHYSICAL & GEOMETRIC PROPERTIES ----------------
% Wave & medium
sound_speed_m_s       = 1500;         % [m/s]
medium_density_kg_m3  = 1000;         % [kg/m^3] (only needed for true intensity)
tone_freq_MHz         = 20;           % [MHz]
freq                   = tone_freq_MHz * 1e6;      % [Hz]
lambda                 = sound_speed_m_s / freq;   % [m]
omega                  = 2*pi*freq;
k0                     = 2*pi/lambda;             % [rad/m]

% Amplitude
amp_Pa                 = 100e3;        % [Pa] peak pressure amplitude (per source, nominal)
% NOTE: k‑Wave source injection vs. ASM ideal monopole have different absolute
% gain. To match absolute values exactly, you may set a calibration factor below.
calibrate_gain         = 1.0;          % Set >1 or <1 to match one reference point if needed.

% Attenuation (alpha_power = 1, no dispersion)
gamma_db_cm_MHz        = 0.5;          % [dB/(cm*MHz)]
use_attenuation        = true;
alpha_db_per_m         = gamma_db_cm_MHz * tone_freq_MHz * 100;  % [dB/m]
alpha_np_per_m         = alpha_db_per_m / 8.685889638;           % [Np/m]
if ~use_attenuation, alpha_np_per_m = 0; end

% Source separations (µm)
sep_min_um             = 0;
sep_step_um            = 5;
sep_max_um             = 2000;
sep_list_um            = sep_min_um:sep_step_um:sep_max_um;
num_separations        = numel(sep_list_um);

% Receiver configuration
receiver_distance_mm   = 0;            % [mm] vertical distance: source -> sensor center
receiver_distance_m    = receiver_distance_mm * 1e-3;

% Sensor geometry: arc centered at (x_ctr, y_ctr)
sensor_radius_m        = 12.7e-3;      % [m] 12.7 mm
sensor_chord_m         = 6.35e-3;      % [m] 6.35 mm
num_arc_points         = 400;          % sampling density along arc

% Domain geometry (match your conventions)
Lx_phys                = 6e-3;         % [m] lateral span, centered at x=0
pml_mm                 = 1.0;          % only for matching your meta (not used by ASM)
src_clearance_mm       = 1.0;

% Grid resolution (match k‑Wave's PPW ~25–40; 3 µm is plenty at 20 MHz)
points_per_w           = 50;           % PPW target; dx = lambda/PPW
ppw_cap                = 120;          % safety cap

% To avoid spectral wrap-around, enforce padding margins beyond the chord
extra_margin_m         = 3.0e-3;       % 1 mm margin each side beyond chord
% You can increase to 2–3 mm if you see edge effects

% Output folder
script_path = mfilename('fullpath'); if isempty(script_path); script_dir = pwd; else; script_dir = fileparts(script_path); end
out_folder = fullfile(script_dir, sprintf('ASM2D_sep_%d_to_%d_step_%d', round(sep_min_um), round(sep_max_um), round(sep_step_um)));
if ~exist(out_folder, 'dir'), mkdir(out_folder); end

%% ----------------------- DERIVED & GRID SETUP ---------------------------
% Lateral grid spacing
dx = lambda / min(max(points_per_w, 1), ppw_cap);  % [m]
dy = dx;                                           % isotropic

% Choose lateral width to cover the sensor chord plus margins (and not smaller than Lx_phys)
Lx = max(Lx_phys, sensor_chord_m + 2*extra_margin_m);
Ly = max( (sensor_radius_m + receiver_distance_m) + extra_margin_m, 4*lambda );  % up to top of arc + margin
            % ensure some minimal height (not critical, ASM propagates per z anyway)

Nx = 2^nextpow2( max(512, ceil(Lx/dx)) );  % FFT-friendly sizes
Ny = 2^nextpow2( max(512, ceil(Ly/dy)) );

% Real-space grids (centered at 0, like your k‑Wave coordinates for x; y starts at 0 upward)
x  = ((1:Nx).' - (Nx+1)/2) * dx;          % column vector, centered at 0
y0 = 0;                                   % define source plane at y = 0 (see note below)

% k-space grids (fftshift ordering)
dkx     = 2*pi / (Nx*dx);
kx      = (-Nx/2:Nx/2-1).' * dkx;         % [rad/m], column
% For 2D (x–y) ASM we only need kx; y is the propagation axis. This is the
% exact 2D Helmholtz propagator for a field evolving along +y.

% Propagator phase root for 2D: kz(kx) = sqrt(k0^2 - kx^2)
% (Evanescent for |kx|>k0 naturally handled; complex sqrt)
kz = sqrt( k0^2 - kx.^2 );                % column


apply_kx_taper = true;   % TEMP diagnostic
kx_norm = abs(kx)/k0;
taper = ones(size(kx));
if apply_kx_taper
    bw = 0.1;  % 10% cosine ramp near cutoff
    idx = kx_norm > (1-bw) & kx_norm < 1;
    taper(idx) = 0.5*(1 + cos(pi*(kx_norm(idx)-(1-bw))/bw));
    taper(kx_norm >= 1) = 0;  % optionally remove evanescent
end


% Print setup
fprintf('ASM2D setup: Nx=%d (dx=%.2f µm, Lx=%.2f mm), λ=%.1f µm @ %.1f MHz\n', ...
    Nx, dx*1e6, (Nx*dx)*1e3, lambda*1e6, tone_freq_MHz);
fprintf('Sensor: R=%.2f mm, chord=%.2f mm, arc points=%d\n', ...
    sensor_radius_m*1e3, sensor_chord_m*1e3, num_arc_points);

%% ----------------------- SENSOR ARC (MATCH k‑Wave) ----------------------
% k‑Wave arc mask uses:
%  - center at (x_ctr, y_ctr = y_src + receiver_distance)
%  - arc on the "upper half" (Yrel >= 0) within +/- theta around vertical
theta = asin(sensor_chord_m / (2*sensor_radius_m));
arc_angles = linspace(pi/2 - theta, pi/2 + theta, num_arc_points).';  % column

% We set the source plane at y = 0.
% If receiver_distance_m > 0, the arc center is shifted by that amount.
arc_ctr_x = 0.0;                                  % center on x=0, as in your plots
arc_ctr_y = y0 + receiver_distance_m;             % sensor center y

% Arc coordinates (absolute positions in meters)
arc_x = arc_ctr_x + sensor_radius_m * cos(arc_angles);
arc_y = arc_ctr_y + sensor_radius_m * sin(arc_angles);

% Arc length coordinate s (0 at top/center)
arc_s = sensor_radius_m * (arc_angles - pi/2);

%% ----------------------- INTENSITY IMAGE CANVAS (for plotting) ----------
% Build an image canvas to mirror your k‑Wave figure. We don’t actually
% compute field everywhere; we place sampled arc values into this image.
Ly_phys_plot = max( arc_y(end) + extra_margin_m, Ny*dy ); % ensure coverage
Ny_plot      = ceil(Ly_phys_plot / dy);
y_plot       = ((1:Ny_plot) - 1).' * dy;           % y >= 0

% Build mapping from physical (x,y) -> nearest grid indices
x_to_idx = @(xm) max(1, min(Nx, round(xm/dx + (Nx+1)/2)));
y_to_idx = @(ym) max(1, min(Ny_plot, round(ym/dy + 1)));

%% ----------------------- MAIN LOOP OVER SEPARATIONS ----------------------
for ii = 1:num_separations
    d_um   = sep_list_um(ii);
    d_m    = d_um * 1e-6;
    half_d = d_m / 2;

    % Two point sources at y=0 (source plane), x = ±half_d
    x1 = -half_d;  y1 = y0;
    x2 = +half_d;  y2 = y0;

    fprintf('[%3.0f µm] Propagating to %d arc points...\n', d_um, num_arc_points);

    % --------------------------------------------------------------------
    % Source spectrum at y=0 (exact sub-voxel point sources):
    % p0(x) = A [ δ(x - x1) + δ(x - x2) ]
    % => P0(kx) = A [ e^{-i kx x1} + e^{-i kx x2} ]
    % --------------------------------------------------------------------
    A0 = calibrate_gain * amp_Pa;  % complex amplitude per source
    P0 = A0 * ( exp(-1i * kx * x1) + exp(-1i * kx * x2) );   % Nx x 1, complex

    % Storage for pressure amplitude on arc
    p_arc_complex = zeros(num_arc_points, 1);

    % --------------------------------------------------------------------
    % For each arc point with coordinate (x_arc(j), y_arc(j)):
    % Propagate P0 by distance z = y_arc(j) - y0, then sample at x_arc(j).
    % --------------------------------------------------------------------
    for jj = 1:num_arc_points
        z_here = arc_y(jj) - y0;    % vertical propagation distance
        if z_here < 0
            % physically not happening with the chosen arc angles; guard anyway
            p_arc_complex(jj) = interp1(x, p_x_z, arc_x(jj), 'pchip', 0);
            continue;
        end

        % Transfer function for this z
        H = exp( 1i * z_here .* kz - alpha_np_per_m * z_here );  % Nx x 1

        % Field at this z, as function of x: p(x, z_here)
        % Mind the fftshift ordering: we constructed kx in shifted order,
        % so inverse transform requires ifftshift.
        Pz_shift = (P0 .* H) .* taper;                       % spectrum at z
        p_x_z    = ifft( ifftshift(Pz_shift) );     % complex field over x

        % Sample at arc_x(jj) via interpolation (outside domain -> 0)
        p_arc_complex(jj) = interp1(x, p_x_z, arc_x(jj), 'linear', 0);
    end

    % Amplitude-based “intensity” quantities (CW, single-frequency)
    p_amp       = abs(p_arc_complex);        % [Pa]
    p_rms       = p_amp/sqrt(2);             % [Pa]
    p_peak      = p_amp;                     % [Pa]
    p_pp        = 2*p_amp;                   % [Pa]
    % If true acoustic intensity in W/m^2 is needed:
    % I_Wm2 = (p_rms.^2) / (medium_density_kg_m3 * sound_speed_m_s);

    %% ------------------- Build “intensity image” like k‑Wave -------------
    % Create a blank image and place the arc pixel values at nearest grid points
    intensity_image = zeros(Nx, Ny_plot);  % will store RMS (active window) equivalent

    % Map arc samples to nearest pixel indices
    arc_x_idx = x_to_idx(arc_x);
    arc_y_idx = y_to_idx(arc_y);

    % Fill with p_rms (k‑Wave "intensity" proxy)
    for jj = 1:num_arc_points
        intensity_image(arc_x_idx(jj), arc_y_idx(jj)) = p_rms(jj);
    end

    % Build the “sensor pixel arrays” like your k‑Wave script
    % (x_mm, y_mm at each arc pixel)
    sensor_x_pos_mm = arc_x * 1e3;
    sensor_y_pos_mm = arc_y * 1e3;

    % Angular positions and arc coordinate identical to k‑Wave math
    Xrel_pix = sensor_x_pos_mm - (arc_ctr_x*1e3);
    Yrel_pix = sensor_y_pos_mm - (arc_ctr_y*1e3);
    sensor_ang = atan2(Yrel_pix, Xrel_pix);                     % [rad]
    sensor_arc_pos_mm = sensor_radius_m * (sensor_ang - pi/2) * 1e3;

    % Sort by x position for the right subplot (like your code)
    [sensor_x_sorted, sort_idx_x] = sort(sensor_x_pos_mm);
    intensity_sorted_x = p_rms(sort_idx_x);

    %% ------------------- PLOT (same structure as your figure) ------------
    fig = figure('Position', [100, 100, 1200, 500], 'Visible', 'off');

    % Subplot 1: Spatial intensity map (image canvas with arc pixels)
    subplot(1,2,1);
    x_coords_mm = x * 1e3;
    y_coords_mm = y_plot * 1e3;   % y >= 0
    imagesc(x_coords_mm, y_coords_mm, intensity_image.');  % transpose for x-y axes as in k‑Wave
    axis image;
    colormap(gca, 'hot');
    cb1 = colorbar; ylabel(cb1, 'Intensity [Pa] (RMS, CW)');
    xlabel('x position [mm]'); ylabel('y position [mm]');
    title(sprintf('Spatial Intensity Map — Separation: %.1f \\mum', d_um));

    hold on;
    % Overlay arc pixels (cyan), as in your k‑Wave figure
    scatter(sensor_x_pos_mm, sensor_y_pos_mm, 10, 'c', 'filled', ...
            'MarkerEdgeColor', 'w', 'MarkerFaceAlpha', 0.6);

    % Overlay source positions (green x) at y=0
    src_x_mm = [x1, x2]*1e3;
    src_y_mm = [y1, y2]*1e3;
    scatter(src_x_mm, src_y_mm, 50, 'g', 'x', 'LineWidth', 2);
    hold off;

    % Subplot 2: Interference pattern along arc vs x
    subplot(1,2,2);
    plot(sensor_x_sorted, intensity_sorted_x, 'b-', 'LineWidth', 2, ...
         'Marker', 'o', 'MarkerSize', 3);
    xlabel('x position along sensor arc [mm]');
    ylabel('Intensity [Pa] (RMS, CW)');
    title(sprintf('Interference Pattern — Separation: %.1f \\mum', d_um));
    grid on;

    % Diagnostic text (same formulas)
    lambda_um = lambda * 1e6;
    d_m       = d_um * 1e-6;
    L_m       = receiver_distance_m;   % center‑to‑center vertical distance
    if L_m == 0
        fringe_spacing_mm = Inf;   % avoid divide-by-zero; sensor arc curvature drives spacing
    else
        fringe_spacing_mm = (lambda * L_m / d_m) * 1e3;
    end
    sensor_arc_span_mm   = sensor_chord_m * 1e3;
    num_fringes_expected = sensor_arc_span_mm / fringe_spacing_mm;
    fraunhofer_criterion = (L_m==0) * Inf + (L_m>0) * ( L_m / (d_m^2 / lambda) );

    diag_text = sprintf(['\\lambda = %.1f \\mum\n' ...
                         'Fringe spacing \\approx %s\n' ...
                         'Arc span = %.2f mm\n' ...
                         'Expected fringes \\approx %s\n' ...
                         'Far-field ratio = %s'], ...
                         lambda_um, ...
                         num2str(fringe_spacing_mm, '%.2f mm'), ...
                         sensor_arc_span_mm, ...
                         num2str(num_fringes_expected, '%.1f'), ...
                         num2str(fraunhofer_criterion, '%.1f'));
    text(0.02, 0.98, diag_text, 'Units','normalized', 'VerticalAlignment','top', ...
         'BackgroundColor','w', 'EdgeColor','k', 'FontSize',9);

    % Save figure
    fig_file = fullfile(out_folder, sprintf('sensor_intensity_%d.png', round(d_um)));
    try
        exportgraphics(fig, fig_file, 'Resolution', 300);
    catch
        saveas(fig, fig_file);
    end
    close(fig);

    %% ------------------- CSV (same column names as your k‑Wave) ----------
    csv_file = fullfile(out_folder, sprintf('sensor_intensity_%d.csv', round(d_um)));
    % Columns: x_mm, y_mm, arc_pos_mm, intensity_rms_window_Pa, intensity_rms_all_Pa,
    %          intensity_pp_Pa, intensity_peak_Pa
    csv_data = [sensor_x_pos_mm, sensor_y_pos_mm, sensor_arc_pos_mm, ...
                p_rms,            p_rms,               p_pp,          p_peak];

    header_line = "x_mm,y_mm,arc_pos_mm,intensity_rms_window_Pa,intensity_rms_all_Pa,intensity_pp_Pa,intensity_peak_Pa\n";
    fid = fopen(csv_file, 'w');
    if fid ~= -1
        fprintf(fid, header_line);
        fclose(fid);
        writematrix(csv_data, csv_file, 'WriteMode','append');
    end

    %% ------------------- MAT (similar content to your k‑Wave save) -------
    mat_file = fullfile(out_folder, sprintf('sensor_data_%d.mat', round(d_um)));
    sensor_data = struct();  % no time series in ASM; keeping key fields for parity
    kgrid = struct('dx', dx, 'dy', dy, 'Nx', Nx, 'Ny', Ny_plot);  
    medium = struct('sound_speed', sound_speed_m_s, 'density', medium_density_kg_m3, ...
                    'alpha_power', 1.0, 'alpha_coeff', gamma_db_cm_MHz, ...
                    'alpha_mode','no_dispersion');
    source = struct(); % in ASM we don’t store time waveform
    sensor = struct(); % no mask array; we store arc arrays below

    sensor_ctr_x_idx = x_to_idx(arc_ctr_x);
    sensor_ctr_y_idx = y_to_idx(arc_ctr_y);

    % Save important arrays/metadata for downstream comparisons
    save(mat_file, 'sensor_data', 'kgrid', 'medium', 'source', 'sensor', ...
         'sensor_ctr_x_idx', 'sensor_ctr_y_idx', ...
         'sensor_radius_m', 'sensor_chord_m', 'dx', 'dy', 'Nx', 'Ny_plot', ...
         'receiver_distance_m', 'tone_freq_MHz', 'amp_Pa', ...
         'd_um', 'p_rms', 'sensor_x_pos_mm', 'sensor_y_pos_mm', ...
         'sensor_arc_pos_mm', 'arc_x', 'arc_y', 'arc_angles', 'arc_s', ...
         'calibrate_gain', 'alpha_np_per_m', 'use_attenuation', 'lambda');

    fprintf('  Saved: %s\n  Saved: %s\n  Saved: %s\n', fig_file, csv_file, mat_file);
end

fprintf('\nAll separations complete. Results in: %s\n', out_folder);
