% V8: Merging 3 lvl and 2 lvl sim

clc; clear all; close all; format compact;
% Constants, experimental and simulation parameters
c = 299792458; % [m/s]
hbar = 1.0545718E-34; % [m^2 kg/s] Planck's constant
kB = 1.38064852E-23; % [m2 kg s-2 K-1] Boltzmann's constant
m = 1.45496E-25; % [kg] Mass of Sr88
%wa = 2.74E15; % [rad/s] Angular freq of 1S0 - 3P1 transition in Sr88
lameg = 689.2585E-9; % [m] Wavelength of 1S0 - 3P1 transition in Sr88 (NIST database)
wa = 2.*pi.*c./lameg; % [rad/s] Angular freq of 1S0 - 3P1 transition in Sr88
gameg = 2.*pi.*7.5E3; % [rad/s] Angular linewidth of Sr88 transition 1S0 - 3P1
gamcg = 2.01E8; % [rad/s] Angular linewidth of Sr88 transition 1S0 - 1P1
gamce = 0; % [rad/s] Modelling decay from 1P1 to 3P1.
eps0 = 8.854187817.*10^-12; % [F/m] Vacuum permittivity

paramGroup = 6; % Parameter group. 0: Appel's niceplot.py. 1: Appel's report. 2: Office poster. 3: Quick test. 4: Best known Sr 1 setup. 5: Sr 1 setup considerations. 6: Strong red pulse
order2 = 0; % If 1, use 2nd order integration, if 0, use 1st order.
redMOT = 0; % With red MOT for Sr 1.
optPulse = 0; % If the script should optimize the pi-pulse during integration. Doesn't work due to new for loops
simRabiPhase = 0; % If the script should modulate pi pulse-atom coupling also by phase
simBfield = 1; % If the script should modulate pi pulse-atom coupling also by B-field
simBlue = 1; % If the script should simulate the blue laser
simTemperature = 1; % If the script should account for non-zero temperature
plots = 1; % If the script should make plots
saveData = 0; % If the sim should save data at end
loadData = 0; % If the sim should load data just before Rabi pulse
simBeforePulse = 1; % If the script should simulate cavity before Rabi pulse
simPulse = 1; % If the script should simulate cavity during/after Rabi pulse
optimize = 0; % If the script should run multiple sims to optimize stuff
optimizeTheta = 0; % If the script should run multiple sims to optimize Rabi pulse beam angle
saveNeDat = 1; % For experiment 2017-02-23 checking Ne(Tpi)
oplotNe = 1; % Overplot number of excited atoms?
oplotP = 1; % Overplot cavity transmission power?
%NeFile = 'C:\Users\blabl\Documents\Bladisk 8G\4-3 Thesis\20170608 Sr 1 Experiments\Data\ExcRate-1.6MHz.mat';
NeFile = 'C:\Users\blabl\Documents\Bladisk 8G\4-3 Thesis\20180313 March Experiments\20180313 Measurements\Excitation Data\dat100.mat';
Pfile = 'C:\Users\blabl\Documents\Bladisk 8G\4-3 Thesis\20170516 Sr 1 Experiments\CTdatMay16.mat';
NICEfile = 'C:\Users\blabl\Documents\Bladisk 8G\4-3 Thesis\20170516 Sr 1 Experiments\NICEdatMay16.mat';
%NeFile = 'C:\Users\Admin\LaserSky\Sr I\Cavity simulation (Mikkel Tang)\ExcRate-1.6MHz.mat';
%Pfile = 'C:\Users\Admin\LaserSky\Sr I\Cavity simulation (Mikkel Tang)\CTdatMay16.mat';
%NICEfile = 'C:\Users\Admin\LaserSky\Sr I\Cavity simulation (Mikkel Tang)\NICEdatMay16.mat';
tPoffset = -36.7E-6; % [s] When pulse starts in power data file
plotNICEfactor = 0; % If the script should use and plot the NICE-OHMS conversion factor for output power
plotCloudExc = 0; % If plotting excitation of the cloud
plotSteps = 200; % Steps to plot atomic cloud
simAOMopen = 0; % If simulating finite opening/close time of pulse AOM

N = 3.0E7; % Number of atoms
Ng = 1.0E4; % Number of groups for atoms
T = 5.0E-3; % [K] Temperature of atoms.
Rxy = 1.2E-3; % [m] Radius of cloud (std) perpendicular to cavity axis
Rz = 1.2E-3; % [m] Radius of cloud (std) along cavity axis
w0 = wa; % [rad/s] Angular freq of cavity mode
detR = 2.*pi.*0E6; % [rad/s] Freq of Rabi pulse detuning wrt cavity mode
dets = -2.*pi.*1.5E6; % [rad/s] Stark shift of atomic energy level due to blue laser
detc = -2.*pi.*40E6; % [rad/s] Jan: -40 MHz
%wp = wa + 2.*pi .* detRabi; % [Hz] Angular freq of Rabi pulse NOT USED BY SIM
wR = 33.2E6; % [Hz] Rabi freq of pi pulse, NOTE: Sometimes calculated further down
R0 = [0 0 0].*1E-3; % [m] Center of atomic cloud
xmot = 2.14E-3; % [m] Atomic cloud center distance from MOT center
L = 0.192; % [m] Cavity length
W = 5E-4; % [m] Cavity mode waist
Wmot = (19./8).*1E-3; % [m] MOT beam waist CURRENTLY NOT IMPLEMENTED
%Wp = [0.5E-3 5.0E-3]; % [m] Pi pulse waists
Wpy = 0.5E-3; % [m] Pi pulse waists along y and xz.
Wpxz = 5.0E-3; % [m] Pi pulse waists along xz.
theta = 0.25 .* pi; % [rad] Pi pulse beam angle with z axis
Pseed = 2.0E-8; % [W] Power of seed laser.
kappa = 2.*pi.*520E3; % [Hz] Cavity lifetime
dt1 = 8E-10; % [s] Timestep during initial blue MOT
dt2 = 50E-10; % [s] Timestep after initial blue MOT
dt3 = 2E-10; % [s] Timestep during Rabi pulse
dt4 = 10E-10; % [s] Timestep after Rabi pulse with blue MOT off
dt5 = 8E-10; % [s] Timestep during final blue MOT
atomRuns = 1; % Number of times the sim should repeat with different number of atoms
dtMOT = 1E-11; % [s] Timestep during MOT on.
tblue1 = 0.0E-6; % [s] Duration of blue cooling light on
tpi = 100.0E-6; % [s] Delay before pi-pulse
tblue2 = 120.0E-6; % [s] When blue laser is turned on after pulse
Tpi = 284E-9; % [s] Pi-pulse duration.
tmax = 120.0E-6; % Simulation time
kAppel = 2.*pi./689e-9; % [1/m] Value of k = w0/c for Appel.
g0 = 4609; % [1/s] Appel
Isat = 0.032; % [W/m^2] Saturation intensity of transition (Jan, Bjarke).
tauAOM = 28e-9; % [s] Characteristic opening/close time of pulse AOM
ntauAOMs = 6; % Number of extra tauAOMs to simulate the Rabi pulse for on each side of the pulse, if simulating AOM opening time: Higher values more accurate

if paramGroup == 1 % Appel's report
    N = 2.0E7; % Number of atoms
    T = 3.0E-3; % [K] Temperature of atoms.
    R = 5.0E-3; % [m] Radius of atomic clump for pos distribution.
    Pseed = 100.0E-9; % [W] Power of seed laser.
    g0 = 2304; % [1/s] Wrongly calculated g0 from Appel report.
elseif paramGroup == 2 % Poster in office
    Pseed = 8E-9; % [W] Power of seed laser.
    Ng = 5.0E4; % Number of groups for atoms
elseif paramGroup == 3 % Quick test
    tpi = 1.0E-6;
    Tpi = 500E-9;
    Ppi = 17.1E-3; % [W] Power of pi pulse laser
    tmax = 3.0E-6;
elseif paramGroup == 4 || paramGroup == 5 || paramGroup == 6 % Best known values for Sr 1 setup
    N = 6E7; % Number of atoms
    Ng = 1.0E4; % Number of groups for atoms
    W = 4.45E-4; % [m] Cavity mode waist
    kappa = 2.*pi.*627E3; % [rad/s] Cavity lifetime
    Pseed = 140E-9; % [W] Power of seed laser.
    Ppi = 20E-3; % [W] Power of pi pulse laser
    %Wp = [2.1 2.1].*1E-3; % [m] Pi pulse waists along y and xz.
    Wpy = 1.73E-3; % [m] Pi pulse waists along y.
    Wpxz = 2.2E-3; % [m] Pi pulse waists along xz.
    %dClump = 5.5E-3; % [m] Diameter of atomic clump where density falls to e^-2 of center (Jan), = 4*std
    %R = dClump ./ 4; % [m] Radius of atomic cloud equivalent to std
    R0 = [0 0 0].*1E-3; % [m] Center of atomic cloud
    Tpi = 2000E-9; % [s] Duration of pi pulse (if not optimizing)
    tblue1 = 23.3E-6; % [s] Duration of blue cooling light on
    tpi = 60.0E-6; % [s] Delay before pi-pulse
    tblue2 = 83.3E-6; % [s] When blue laser is turned on after pulse
    tmax = 90E-6;
    if paramGroup == 5 || paramGroup == 6 % Sr 1 setup considerations
        N = 6E7;
        Ppi = 22E-3; %22E-3; % [W] Power of pi pulse laser
        %Wpy = 2.8E-3; % [m] Pi pulse waists along y and xz.
        %Wpxz = 2.8E-3; % [m] Pi pulse waists along xz.
        T = 3.9E-3; % [K] Temperature of atoms.
        Tpi = 2000E-9; % [s] Duration of pi pulse (if not optimizing)
        Rxy = 1.15E-3; % [m] Radius of cloud (std) perpendicular to cavity axis
        Rz = 1.15E-3; % [m] Radius of cloud (std) along cavity axis
        %xmot = 2.14E-3; % [m] Atomic cloud center distance from MOT center
        xmot = 4E-3; % Trying to get right excitation rate
        %Gamma = 0; % Jan: Check Gamma = 0
        if paramGroup == 6
            Ppi = 113e-3;
            Wpy = 1.65E-3;
            Wpxz = 1.48E-3;
            Rxy = 0.9E-3; % [m] Radius of cloud (std) perpendicular to cavity axis
            Rz = 0.9E-3; % [m] Radius of cloud (std) along cavity axis
            xmot = 4E-3;
            T = 5.0E-3;
        end
    end
    if redMOT == 1
        Rxy = Rxy ./ 3; % [m] Radius of cloud (std) perpendicular to cavity axis
        Rz = Rz ./ 3; % [m] Radius of cloud (std) along cavity axis
        T = 100E-6;
    end
    g0 = sqrt(6.*c.^3 .* gameg .* w0 ./ (W.^2 .* L .* wa.^3)); % [1/s] Unmodulated atom-cav coupling strength.
    g0blue = sqrt(6.*c.^3 .* gameg .* w0 ./ (W.^2 .* L .* (wa + dets).^3)); % [1/s] Unmodulated atom-cav coupling strength (Stark shift).
    wR = gameg .* sqrt(Ppi ./ (4.*Isat.*pi.*Wpy.*Wpxz)); % Rabi freq at cloud center (notes p. 80-81), ~agrees with Jan
    %wR = 9.2E6; % [Hz] Rabi freq of pi-pulse (Jan)
    TpiCalc = pi ./ wR; % Save calculated pi pulse length for comparison
end

sigN = 0.2*N; % Standard deviation in number of atoms
Narr = abs(N+sigN.*randn(atomRuns,1));

if simTemperature == 0
    T = 0;
end

% Optimization of pi pulse
nTheta = 1; % Spacing in laser angle
nWpy = 1; % Spacing in pulse waist perpendicular to mode
nWpxz = 1; % Spacing in pulse waist along mode plane
thetaMin = 0.05.*pi; % Min laser angle
thetaMax = 0.5.*pi; % Max laser angle
WpyMin = 1.5E-3; % [m] Min waist
WpyMax = 4E-3; % [m] Max waist
WpxzMin = 1.5E-3; % [m] Min waist
WpxzMax = 5E-3; % [m] Max waist
if optimize == 0
    nTheta = 1; nWpy = 1; nWpxz = 1;
    WpyMin = Wpy; WpyMax = Wpy;
    WpxzMin = Wpxz; WpxzMax = Wpxz;
    thetaMin = theta; thetaMax = theta;
elseif optimizeTheta == 1
    nWpy = 1; nWpxz = 1;
    WpyMin = Wpy; WpyMax = Wpy; WpxzMin = Wpxz; WpxzMax = Wpxz;
end

optAttempts = nTheta .* nWpy .* nWpxz; % Number of attempts to optimize parameters
thetas = linspace(thetaMin,thetaMax,nTheta); % Array of angles
Wpys = linspace(WpyMin,WpyMax,nWpy); % Array of y waists
Wpxzs = linspace(WpxzMin,WpxzMax,nWpxz); % Array of xz waists

if simAOMopen == 1 % So the simulation covers the lower pi pulse powers
    Tpi = Tpi+2.*ntauAOMs.*tauAOM;
end

tsteps1 = round(tblue1./dt1+1); % Number of time steps in simulation with MOT on initially
tsteps2 = tsteps1+round((tpi-tblue1)./dt2+1); % Number of time steps in simulation while MOT off before pulse
tsteps3 = tsteps2+round(Tpi./dt3+1); % Number of time steps in simulation during pi pulse
tsteps4 = tsteps3+round((tblue2-tpi-Tpi)./dt4+1); % Number of time steps in simulation after pi pulse with MOT off
tsteps = tsteps4+round((tmax-tblue2)./dt5); % Number of time steps in simulation during final MOT on

% Calculations
eta = sqrt(Pseed .* kappa ./ (hbar .* w0)); % Eta for steady state empty cavity.
if simAOMopen == 1 % Array of modulated pi pulse powers
    PpiStep = zeros(1,tsteps3-tsteps2);
end

% Simulation arrays and initial conditions
t = zeros(tsteps,1); % Time array
r = zeros(Ng,3,atomRuns); % Initial position array
v = zeros(Ng,3,atomRuns); % Velocity array
f = zeros(Ng,atomRuns); % Position modulation factor for g
h = zeros(Ng,atomRuns); % Velocity modulation factor for g
g = zeros(Ng,atomRuns); % Modulated atom-cavity coupling strength
Pgg = (N./Ng).*ones(Ng,atomRuns); % Density matrix element
Pee = zeros(Ng,atomRuns); % Density matrix element
Pge = zeros(Ng,atomRuns); % S_minus
Sz = -(N./Ng).*ones(Ng,atomRuns); % S_z
A = zeros(tsteps,atomRuns); % Photon annihilation operator expectation value
n = zeros(tsteps,atomRuns); % Photon number operator expectation value
Pout = zeros(tsteps,atomRuns); % [W] Output power
Ne = zeros(tsteps,atomRuns); % Number of excited atoms
coh = zeros(tsteps,atomRuns); % Atomic coherence
A(1,:) = - 1i.*eta ./ kappa;
n(1,:) = eta.^2 ./ kappa.^2;

PggPlot = zeros(Ng,plotSteps); % For plotting exc rate
PeePlot = zeros(Ng,plotSteps); % For plotting exc rate
PgePlot = zeros(Ng,plotSteps); % For plotting exc rate
rPlot = zeros(Ng,3,plotSteps); % For plotting exc rate
tPlot = zeros(plotSteps,1);
PpiStepPlot = zeros(plotSteps,1);

% Gaussian initial distribution of atoms: R is std, R0 is center.
r = bsxfun(@plus,R0,randn(Ng,3,atomRuns)); % Position array, normalized. Below multiplied in stds to get asymmetric atomic cloud.
r(:,1,:) = r(:,1,:).* Rxy; r(:,2,:) = r(:,2,:) .* Rxy; r(:,3,:) = r(:,3,:) .* Rz;

for grp = 1:1:Ng
    % Maxwell-Boltzmann distribution of atom velocities
    v(grp,1,:) = sqrt(kB.*T./m).*randn(1,atomRuns);
    v(grp,2,:) = sqrt(kB.*T./m).*randn(1,atomRuns);
    v(grp,3,:) = sqrt(kB.*T./m).*randn(1,atomRuns);
    if tblue1 > 0 && simBlue == 1
        h(grp,:) = 1 ./ (1 + (4./kappa.^2).*(w0.*v(grp,3,:)./c + dets).^2); % Velocity and Stark modulation of g
        f(grp,:) = sin(2.*pi.*rand(1,atomRuns)); % Position modulation of g - completely random phase due to blue laser
        %f(grp,:) = sin((w0./c) .* (r(grp,3,:) + v(grp,3,:).*t(1))); % Position modulation of g
        g(grp,:) = g0blue.*f(grp,:).*h(grp,:).*exp(-(squeeze(r(grp,1,:))'.^2 + squeeze(r(grp,2,:))'.^2)./W.^2); % Modulation including Gaussian beam during Stark shift
    else
        h(grp,:) = 1 ./ (1 + 4.*(w0.*v(grp,3,:)./(c.*kappa)).^2); % Velocity modulation of g
        f(grp,:) = sin((w0./c) .* (r(grp,3,:) + v(grp,3,:).*t(1))); % Position modulation of g
        g(grp,:) = g0.*f(grp,:).*h(grp,:).*exp(-(squeeze(r(grp,1,:))'.^2 + squeeze(r(grp,2,:))'.^2)./W.^2); % Modulation including Gaussian beam
    end
end
coh = mean(abs(sum(Pge.*sign(g)))); % Initial atomic coherence
vAtomPulse = cos(theta).*v(:,3,:) + sin(theta).*v(:,1,:); % Speed of atoms wrt pi-pulse
Dwpa = -vAtomPulse.*kAppel + detR; % [rad/s] wp-wa, detuning of pi-pulse due to atom velocities and adjusted detuning

%% Numeric integration over time
disp('Running integration...'); % Display start of integration
% Definitions for efficiency
Npg = N./Ng; % Atoms per group
etaD2 = 0.5.*eta; % Half eta
kapD2 = 0.5.*kappa; % Half kappa
GamD2 = 0.5.*gameg; % Half Gamma
kapSqD4 = 0.25.*kappa.^2; % Quarter of kappa squared
GamSqD4 = 0.25.*gameg.^2; % Quarter of Gamma squared

Omegc0 = 2.5E8; % [rad/s] GUESS

tic % Starts timer
if simBeforePulse == 1
disp('Now simulating initial MOT on...');
dt = dt1;
for step = 2:1:tsteps1 % Initial MOT on
    t(step) = t(step-1) + dt; % Add time step
    r = r + v.*dt; % Update position
    %f = sin(2.*pi.*rand(Ng,1)); % Position modulation of g - completely random phase due to blue laser
    f = sin(kAppel .* r(:,3,:)); % Position modulation of g
    g = g0blue.*squeeze(f).*h.*exp(-(squeeze(r(:,1,:)).^2 + squeeze(r(:,2,:)).^2)./W.^2); % Gaussian beam
    
    A(step,:) = A(step-1,:) + dt.*(-1i.*sum(g.*Pge) - 1i.*etaD2 - kapD2.*A(step-1,:));
    for kAtom = 1:1:atomRuns
        Omegs(:,kAtom) = g(:,kAtom).*A(step,kAtom);
    end

    % Equations from 20170830
    Pee = Pee + dt.*(1i.*conj(Omegs).*Pge - 1i.*Omegs.*conj(Pge) - gameg.*Pee);
    Pgg = Npg - Pee;
    Pge = Pge + dt.*((-0.5.*gameg-1i.*dets).*Pge + 1i.*Omegs.*(Pee-Pgg));
    
    Pout(step) = mean(conj(A(step,:)).*A(step,:) .* kappa .* hbar .* w0); % [W] Output power from photon number
    popg(step) = mean(sum(Pgg));
    pope(step) = mean(sum(Pee));
    coh(step) = mean(abs(sum(Pge.*sign(g)))); % Atomic coherence

    if rem(step,5000) == 0
        disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s.']); % Display how the sim is progressing
    end
end
disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s. Now simulating initial MOT off...']); % Display how the sim is progressing
dt = dt2;
h = squeeze(1 ./ (1 + 4.*(w0.*v(:,3,:)./(c.*kappa)).^2)); % Velocity modulation of g
for step = tsteps1+1:1:tsteps2 % After initial MOT, before pi pulse
    t(step) = t(step-1) + dt; % Add time step
    r = r + v.*dt; % Update position
    f = sin(kAppel .* r(:,3,:)); % Position modulation of g
    g = g0.*squeeze(f).*h.*exp(-(squeeze(r(:,1,:)).^2 + squeeze(r(:,2,:)).^2)./W.^2); % Gaussian beam

    A(step,:) = A(step-1,:) + dt.*(-1i.*sum(g.*Pge) - 1i.*etaD2 - kapD2.*A(step-1,:));
    for kAtom = 1:1:atomRuns
        Omegs(:,kAtom) = g(:,kAtom).*A(step,kAtom);
    end

    Pee = Pee + dt.*(1i.*conj(Omegs).*Pge - 1i.*Omegs.*conj(Pge) - gameg.*Pee);
    Pgg = Npg - Pee;
    Pge = Pge + dt.*(-0.5.*gameg.*Pge + 1i.*Omegs.*(Pee-Pgg) - 0i.*dets.*Pge);

    Pout(step) = mean(conj(A(step,:)).*A(step,:) .* kappa .* hbar .* w0); % [W] Output power from photon number
    popg(step) = mean(sum(Pgg));
    pope(step) = mean(sum(Pee));
    coh(step) = mean(abs(sum(Pge.*sign(g)))); % Atomic coherence

    if rem(step,5000) == 0
        disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s.']); % Display how the sim is progressing
    end
end

else
    t(2:tsteps1) = linspace(2.*dt,tsteps1.*dt,tsteps1-1);
end

optStep = 0;
nAll = zeros(length(n),nTheta,nWpy,nWpxz); cohAll = nAll; popeAll = nAll;
TpiAll = zeros(nTheta,nWpy,nWpxz);
for indTheta = 1:1:nTheta % Optimize laser angle
for indWpy = 1:1:nWpy % Optimize cavity waist y
for indWpxz = 1:1:nWpxz % Optimize cavity waist xz
    if optimize == 1 % If optimizing, set parameters from the optimization ranges
        Wpy = Wpys(indWpy);
        Wpxz = Wpxzs(indWpxz);
        wR = gameg .* sqrt(Ppi ./ (4.*Isat.*pi.*Wpy.*Wpxz)); % Recalculate Rabi freq at cloud center
        theta = thetas(indTheta);
        optStep = optStep + 1;
        disp(['Running optimization ',num2str(optStep),' of ',num2str(optAttempts)]); % Display progress with optimizations
    end
    if loadData == 1 || optimize == 1 % Load simulation data
        dat1 = importdata('ensembleDat.txt');
        Ne = dat1.data(:,1);
        coh = dat1.data(:,2);
        A = dat1.data(:,3)+1i.*dat1.data(:,4);
        dat2 = importdata('atomDat.txt');
        r = [dat2.data(:,1) dat2.data(:,2) dat2.data(:,3)];
        v = [dat2.data(:,4) dat2.data(:,5) dat2.data(:,6)];
        Sz = dat2.data(:,7);
        Pge = dat2.data(:,8)+1i.*dat2.data(:,9);
        
        % Calculations
        if simTemperature == 0
            T = 0;
            v = 0.*v;
        end
        h = squeeze(1 ./ (1 + 4.*(w0.*v(:,3,:)./(c.*kappa)).^2)); % Velocity modulation of g
        vAtomPulse = cos(theta).*v(:,3) + sin(theta).*v(:,1); % Speed of atoms wrt pi-pulse
        Dwpa = -vAtomPulse.*kAppel + detR; % [rad/s] wp-wa, detuning of pi-pulse due to atom velocities and adjusted detuning
    end
    if optPulse == 1
        Tpi = 10E6; % Set pi-pulse very long and change it back later during integration.
    end
    if simPulse == 1 % Simulate Rabi-pulse
    disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s. Now simulating Rabi pulse...']); % Display how the sim is progressing
    dt = dt3;
    pltStep = 1; % Step number in animation of atomic cloud excitation
    pltStepSpacing = ceil((tsteps3-tsteps2)./plotSteps);
    for step = tsteps2+1:1:tsteps3 % During Rabi pulse
        if rem(step,pltStepSpacing) == 0
            if plotCloudExc == 1
                PeePlot(:,pltStep) = min(max(Pee(:,1),zeros(Ng,1)),Npg.*ones(Ng,1));
                PgePlot(:,pltStep) = Pge(:,1);
                rPlot(:,:,pltStep) = r(:,:,1);
                tPlot(pltStep) = t(step-1);
                PpiStepPlot(pltStep) = PpiStep(step-tsteps2-1);
                pltStep = pltStep + 1;
            end
        end
        
        t(step) = t(step-1) + dt; % Add time step
        r = r + v.*dt; % Update position
        f = sin(kAppel .* r(:,3,:)); % Position modulation of g
        g = g0.*squeeze(f).*h.*exp(-(squeeze(r(:,1,:)).^2 + squeeze(r(:,2,:)).^2)./W.^2); % Gaussian beam
        %wRs = wR .* exp(- squeeze(r(:,2,:)).^2 ./ Wpy.^2 - (squeeze(r(:,1,:)).^2 .*cos(theta).^2 + squeeze(r(:,3,:)).^2 .* sin(theta).^2) ./ Wpxz.^2); % Gauss I modulation
        if simAOMopen == 1 % Simulate pulse AOM opening time
            PpiStep(step-tsteps2) = Ppi./...
                ((1+exp((t(step)-(tpi+Tpi-ntauAOMs.*tauAOM))/tauAOM))...
                .*(1+exp(-(t(step)-(tpi+ntauAOMs.*tauAOM))/tauAOM))); % [W] Modulated pi pulse power
            wR = gameg .* sqrt(PpiStep(step-tsteps2) ./ (4.*Isat.*pi.*Wpy.*Wpxz)); % Rabi freq at cloud center
        end
        wRs = wR .* exp(- squeeze(r(:,2,:)).^2 ./ Wpy.^2 - (squeeze(r(:,1,:)) - squeeze(r(:,3,:))).^2 ./ Wpxz.^2); % Gauss I modulation
        if simBfield == 1
            wRs = wRs .* (4.*(squeeze(r(:,1,:)) - xmot).^2 ./ (4.*(squeeze(r(:,1,:)) - xmot).^2 + squeeze(r(:,2,:)).^2 + squeeze(r(:,3,:)).^2)); % B-field I modulation
        end
        if simRabiPhase == 1
            wRs = wRs .* exp(1i .* kAppel .* (squeeze(r(:,1,:)).*sin(theta) + squeeze(r(:,3,:)).*cos(theta))); % Beam phase modulation
        end
        wRsD2 = 0.5.*wRs; % Half Rabi frequency for calc efficiency

        A(step,:) = A(step-1,:) + dt.*(-1i.*sum(g.*Pge) - 1i.*etaD2 - kapD2.*A(step-1,:));
        for kAtom = 1:1:atomRuns
            Omegs(:,kAtom) = g(:,kAtom).*A(step,kAtom);
        end

        % Equations before 20180125
%         Pee = Pee + dt.*(1i.*conj(Omegs./g).*Pge + 0.5i.*conj(wRs).*Pge - 1i.*(Omegs./g).*conj(Pge) - 0.5i.*wRs.*conj(Pge) - gameg.*Pee);
%         Pgg = Npg - Pee;
%         Pge = Pge + dt.*(-0.5.*gameg.*Pge + 1i.*Omegs.*(Pee-Pgg) - 1i.*detR.*Pge + 0.5i.*wRs.*(Pee-Pgg));
         
        % Equations from 20180125, made more similar to Sr1_V7
        Pee = Pee + dt.*(2.*imag(A(step,:).*conj(Pge)) - imag(Pge.*wRs) - gameg.*Pee);
        Pgg = Npg - Pee;
        %Pge = Pge + dt.*(-0.5.*gameg.*Pge + 1i.*Omegs.*(Pee-Pgg) - 1i.*detR.*Pge + 0.5i.*wRs.*(Pee-Pgg));
        Pge = Pge + dt.*(-0.5.*gameg.*Pge + 1i.*Omegs.*(Pee-Pgg) - 1i.*Dwpa.*Pge + 0.5i.*wRs.*(Pee-Pgg));
        
%         if step < tsteps2+5
%             disp(['step ',num2str(step-tsteps2),' dPee/dt: ']);
%             disp(mean(-gameg.*Pee));
%             disp(mean(2.*imag(A(step,:).*conj(Pge))));
%             disp(mean(-imag(Pge.*wRs)));
%             disp('dPge/dt:');
%             disp(mean(-0.5.*gameg.*Pge));
%             disp(mean(1i.*Omegs.*(Pee-Pgg)));
%             disp(mean(0.5i.*wRs.*(Pee-Pgg)));
%             disp(mean(-1i.*detR.*Pge));
%         end
        
        Pout(step) = mean(conj(A(step,:)).*A(step,:) .* kappa .* hbar .* w0); % [W] Output power from photon number
        popg(step) = mean(sum(Pgg));
        pope(step) = mean(sum(Pee));
        coh(step) = mean(abs(sum(Pge.*sign(g)))); % Atomic coherence
        
        if optPulse == 1 % Stop applying pi-pulse if excited pop reaches a max
            if pope(step) <= pope(step-1)
                Tpi = t(step-1) - tpi; % Save length of pulse when max Ne was reached
            end
        end

        if rem(step,5000) == 0
            disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s.']); % Display how the sim is progressing
        end
    end
    end
    disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s. Now simulating after Rabi pulse while MOT off...']); % Display how the sim is progressing
    dt = dt4;
    for step = tsteps3+1:1:tsteps4 % After Rabi pulse, before MOT
        t(step) = t(step-1) + dt; % Add time step
        r = r + v.*dt; % Update position
        f = sin(kAppel .* r(:,3,:)); % Position modulation of g
        g = g0.*squeeze(f).*h.*exp(-(squeeze(r(:,1,:)).^2 + squeeze(r(:,2,:)).^2)./W.^2); % Gaussian beam
        
        A(step,:) = A(step-1,:) + dt.*(-1i.*sum(g.*Pge) - 1i.*etaD2 - kapD2.*A(step-1,:));
        for kAtom = 1:1:atomRuns
            Omegs(:,kAtom) = g(:,kAtom).*A(step,kAtom);
        end

        Pee = Pee + dt.*(1i.*conj(Omegs).*Pge - 1i.*Omegs.*conj(Pge) - gameg.*Pee);
        Pgg = Npg - Pee;
        Pge = Pge + dt.*(-0.5.*gameg.*Pge + 1i.*Omegs.*(Pee-Pgg));
        
        Pout(step) = mean(conj(A(step,:)).*A(step,:) .* kappa .* hbar .* w0); % [W] Output power from photon number
        popg(step) = mean(sum(Pgg));
        pope(step) = mean(sum(Pee));
        coh(step) = mean(abs(sum(Pge.*sign(g)))); % Atomic coherence

        if rem(step,5000) == 0
            disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s.']); % Display how the sim is progressing
        end
    end
    disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s. Now simulating after Rabi pulse while MOT on...']); % Display how the sim is progressing
    dt = dt5;
    h = squeeze(1 ./ (1 + (4./kappa.^2).*(w0.*v(:,3,:)./c + dets).^2)); % Velocity and Stark modulation of g
    for step = tsteps4+1:1:tsteps % After Rabi pulse during MOT
        t(step) = t(step-1) + dt; % Add time step
        r = r + v.*dt; % Update position
        %f = sin(2.*pi.*rand(Ng,1)); % Position modulation of g - completely random phase due to blue laser
        f = sin(kAppel .* r(:,3,:)); % Position modulation of g
        g = g0blue.*squeeze(f).*h.*exp(-(squeeze(r(:,1,:)).^2 + squeeze(r(:,2,:)).^2)./W.^2); % Gaussian beam
        
        A(step,:) = A(step-1,:) + dt.*(-1i.*sum(g.*Pge) - 1i.*etaD2 - kapD2.*A(step-1,:));
        for kAtom = 1:1:atomRuns
            Omegs(:,kAtom) = g(:,kAtom).*A(step,kAtom);
        end

        Pee = Pee + dt.*(1i.*conj(Omegs).*Pge - 1i.*Omegs.*conj(Pge) - gameg.*Pee);
        Pgg = Npg - Pee;
        Pge = Pge + dt.*((-0.5.*gameg-1i.*dets).*Pge + 1i.*Omegs.*(Pee-Pgg));

        Pout(step) = mean(conj(A(step,:)).*A(step,:) .* kappa .* hbar .* w0); % [W] Output power from photon number
        popg(step) = mean(sum(Pgg));
        pope(step) = mean(sum(Pee));
        coh(step) = mean(abs(sum(Pge.*sign(g)))); % Atomic coherence

        if rem(step,5000) == 0
            disp([sprintf('%0.2f',100.*step./tsteps),' % done in ',sprintf('%0.2f',toc),' s.']); % Display how the sim is progressing
        end
    end
    
    nAll(:,indTheta,indWpy,indWpxz) = mean(conj(A).*A,2);
    cohAll(:,indTheta,indWpy,indWpxz) = mean(coh,2);
    popeAll(:,indTheta,indWpy,indWpxz) = pope';
    TpiAll(indTheta,indWpy,indWpxz) = Tpi;
end
end
end
disp([sprintf('Finished simulation.')]); % Display how the sim is progressing

% Determine optimal pi-pulse for pulse intensity
nAllSize=size(nAll); % get the size
nAllMax=max(max(max(nAll(tsteps1:end,:,:,:)))); % find the global maximum
nAllMaxPos=find(nAll(:)==nAllMax); % find the position(s) of the maximum (maxima)
[nWinStep,nWinTheta,nWinWpy,nWinWpxz]=ind2sub(nAllSize,nAllMaxPos); % convert the linear index to 4D indices
nIndWin = [nWinStep,nWinTheta,nWinWpy,nWinWpxz];
%nAll(nWinStep,nWinTheta,nWinWpy,nWinWpxz) % that's the maximum

% Determine optimal pi-pulse for excitation
NeAllSize=size(popeAll); % get the size
NeAllMax=max(max(max(popeAll(tsteps1:end,:,:,:)))); % find the global maximum
NeAllMaxPos=find(popeAll(:)==NeAllMax); % find the position(s) of the maximum (maxima)
[NeWinStep,NeWinTheta,NeWinWpy,NeWinWpxz]=ind2sub(NeAllSize,NeAllMaxPos); % convert the linear index to 4D indices
NeIndWin = [NeWinStep,NeWinTheta,NeWinWpy,NeWinWpxz];
%NeAll(NeWinStep,NeWinTheta,NeWinWpy,NeWinWpxz) % that's the maximum

% Display results
format shortG;
if simPulse == 1
    if optPulse == 1
        disp(['Rabi pulse duration [ns] = ',num2str(1E9.*Tpi),' \pm ',num2str(1E9.*dt./2),' vs ',num2str(1E9.*TpiCalc)]);
    else
        disp(['Rabi pulse duration [ns] = ',num2str(1E9.*Tpi)]);
    end
end
if optimize == 1
    disp(['Max n in pulse = ',num2str(round(nAll(nWinStep,nWinTheta,nWinWpy,nWinWpxz))),' for \theta = ',num2str(thetas(nWinTheta)),...
          ', Wpy = ',num2str(1E3.*Wpys(nWinWpy)),' mm, Wpxz = ',num2str(1E3.*Wpxzs(nWinWpxz)),' mm.']);
    disp(['Max excited fraction = ',num2str(popeAll(NeWinStep,NeWinTheta,NeWinWpy,NeWinWpxz)./N),' for \theta = ',num2str(thetas(NeWinTheta)),...
          ', Wpy = ',num2str(1E3.*Wpys(NeWinWpy)),' mm, Wpxz = ',num2str(1E3.*Wpxzs(NeWinWpxz)),' mm.']);
else
    disp(['Max n in pulse = ',num2str(round(max(n(tsteps1:length(n)))))]);
    disp(['Max excited fraction = ',num2str(max(Ne(tsteps1:length(Ne)))./N)]);
end
disp(['Max output power [nW] = ',num2str(max(1E9.*Pout(tsteps1:length(Pout))))]);
format short;

%% Save simulation data
if saveData == 1
    disp('Saving simulation data.');
    [fileID,errmsg] = fopen('ensembleDat.txt','w');
    fprintf(fileID,'%16s %16s %16s %16s\n','Ne','coh','Re(a)','Im(A)');
    fprintf(fileID,'%16.8E %16.8E %16.8E %16.8E \n',[Ne coh real(A) imag(A)]');
    fclose(fileID);
    [fileID,errmsg] = fopen('atomDat.txt','w');
    fprintf(fileID,'%16s %16s %16s %16s %16s %16s %16s %16s %16s \n','r_x','r_y','r_z','v_x','v_y','v_z','Sz','Re(Sm)','Im(Sm)');
    fprintf(fileID,'%16.8E %16.8E %16.8E %16.8E %16.8E %16.8E %16.8E %16.8E %16.8E \n',[r(:,1) r(:,2) r(:,3) v(:,1) v(:,2) v(:,3) Sz real(Pge) imag(Pge)]');
    fclose(fileID);
    disp('Done.');
end

%% Include NICE-OHMS signal to predict Pout
if plotNICEfactor == 1
    niceData = importdata(NICEfile);
    NICEpoutFactor = cos((niceData(:,2)+0.008)./0.015).^2;
    figure();
    hold on;
    plot(niceData(:,1),NICEpoutFactor,'k-');
    plot(niceData(:,1),niceData(:,2),'r-');
    hold off;
else
    NICEpoutFactor = 1;
end

%% Plots

if plots == 1
    if  optimize == 0
        figure('units','normalized','outerposition',[0 0 1 1]); % Excited atoms
        hold on;
        set(gca,'FontSize',15);
        %ymax = 100;
        ymax = real(102.*max(pope)./N);
        plot(1E6.*t(1:1:length(t)),100.*pope(1:1:length(pope))./N,'r-','LineWidth',2);
        line(1E6.*[tpi tpi], [0 ymax], 'Color',[0 0.5 0],'LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 ymax], 'Color',[0 0 0],'LineWidth',2); % Plot line for end of pi-pulse
        axis([0 1E6.*max(t) 0 ymax]);
        xlabel('Time [탎]'); ylabel('^3P_1 Population [%]'); grid on;
        legend('^3P_1','Pulse on','Pulse off','Location','NorthWest');
        hold off;

        figure('units','normalized','outerposition',[0 0 1 1]); % Atomic coherence
        plot(1E6.*t(1:1:length(t)),coh(1:1:length(coh)),'r-');
        line(1E6.*[tpi tpi], [0 max(coh(1:1:length(coh)))], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 max(coh(1:1:length(coh)))], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
        axis([0 1E6.*max(t) 0 max(coh)]);
        xlabel('Time [탎]'); ylabel('coh'); grid on;

%         figure('units','normalized','outerposition',[0 0 1 1]); % Cavity photons
%         plot(1E6.*t(1:1:length(t)),n(1:1:length(n)),'r-');
%         line(1E6.*[tpi tpi], [0 max(n(1:1:length(n)))], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
%         line(1E6.*[tpi+Tpi tpi+Tpi], [0 max(n(1:1:length(n)))], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
%         axis([0 1E6.*max(t) 0 max(n)]);
%         xlabel('Time [탎]'); ylabel('n'); grid on;
        
        figure('units','normalized','outerposition',[0 0 1 1]); % Cavity transmission
        hold on;
        set(gca,'FontSize',15);
        PseedArr = Pseed.*ones(length(t),1);
        plot(1E6.*t(1:1:length(t)),1E9.*PseedArr,'m--','LineWidth',2);
        plot(1E6.*t(1:1:length(t)),1E9.*Pout(1:1:length(n)),'r-','LineWidth',2);
        if oplotP == 1
            import = importdata(Pfile);
            %plot(1E6.*(import(:,1)+tpi+tPoffset),import(:,2),'b-','LineWidth',2);
            %ymaxCT = 1.02.*max([max(import(:,2)) max(1E9.*Pout)]);
            plot(1E6.*(import(:,1)+tpi+tPoffset),import(:,2)./NICEpoutFactor,'b-','LineWidth',2);
            ymaxCT = 1.02.*max([max(import(:,2)./NICEpoutFactor) max(1E9.*Pout)]);
        end
        axis([0 1E6.*max(t) 0 ymaxCT]);
        line(1E6.*[tpi tpi], [0 ymaxCT], 'Color',[0 0.5 0],'LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 ymaxCT], 'Color',[0 0 0],'LineWidth',2); % Plot line for end of pi-pulse
        if simBlue == 1
            line(1E6.*[tblue1 tblue1], [0 ymaxCT], 'Color',[0 0.5 1],'LineWidth',2); % Plot line for blue laser toggle
            line(1E6.*[tblue2 tblue2], [0 ymaxCT], 'Color',[0 0 1],'LineWidth',2); % Plot line for blue laser toggle
            if oplotP == 1
                legend('P_{seed}','Simulation','Experiment','Pulse start','Pulse end','MOT off','MOT on','Location','NorthWest');
            else
                legend('P_{seed}','Simulation','Pulse start','Pulse end','MOT off','MOT on','Location','NorthWest');
            end
        else
            if oplotP == 1
                legend('P_{seed}','Simulation','Experiment','Pulse start','Pulse end','Location','NorthWest');
            else
                legend('P_{seed}','Simulation','Pulse start','Pulse end','Location','NorthWest');
            end
        end
        xlabel('Time [탎]'); ylabel('P_{out} [nW]'); grid on; hold off;
        
    else
        cmap = hsv(optAttempts); % Generate color map
        
        figure('units','normalized','outerposition',[0 0 1 1]); % Excited atoms
        hold on; indPlot = 1;
        for indTheta = 1:1:nTheta
        for indWpy = 1:1:nWpy
        for indWpxz = 1:1:nWpxz
            plot(1E6.*t,popeAll(:,indTheta,indWpy,indWpxz)./N,'Color',cmap(indPlot,:));
            indPlot = indPlot + 1;
        end
        end
        end
        line(1E6.*[tpi tpi], [0 max(Ne(1:1:length(Ne)))./N], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 max(Ne(1:1:length(Ne)))./N], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
        axis([0 1E6.*max(t) 0 max(max(max(max(popeAll))))./N]);
        xlabel('Time [탎]'); ylabel('N_e / N'); grid on;

        figure('units','normalized','outerposition',[0 0 1 1]); % Atomic coherence
        hold on; indPlot = 1;
        for indTheta = 1:1:nTheta
        for indWpy = 1:1:nWpy
        for indWpxz = 1:1:nWpxz
            plot(1E6.*t,cohAll(:,indTheta,indWpy,indWpxz),'Color',cmap(indPlot,:));
            indPlot = indPlot + 1;
        end
        end
        end
        line(1E6.*[tpi tpi], [0 max(max(max(max(cohAll))))], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 max(max(max(max(cohAll))))], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
        axis([0 1E6.*max(t) 0 max(max(max(max(cohAll))))]);
        xlabel('Time [탎]'); ylabel('coh'); grid on;

        figure('units','normalized','outerposition',[0 0 1 1]); % Cavity photons
        hold on; indPlot = 1;
        for indTheta = 1:1:nTheta
        for indWpy = 1:1:nWpy
        for indWpxz = 1:1:nWpxz
            plot(1E6.*t,nAll(:,indTheta,indWpy,indWpxz),'Color',cmap(indPlot,:));
            indPlot = indPlot + 1;
        end
        end
        end
        line(1E6.*[tpi tpi], [0 max(max(max(max(nAll))))], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 max(max(max(max(nAll))))], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
        axis([0 1E6.*max(t) 0 max(max(max(max(nAll))))]);
        xlabel('Time [탎]'); ylabel('n'); grid on;
        
        nMaxAll = max(nAll(tsteps1:end,:,:,:)); % Maxima for all optimization parameters
        nMaxAllSq = squeeze(nMaxAll);
        NeMaxAll = max(popeAll(tsteps1:end,:,:,:)); % Maxima for all optimization parameters
        NeMaxAllSq = squeeze(NeMaxAll);
        
        if optimizeTheta == 0
            % Contour plot for n
            figure('units','normalized','outerposition',[0 0 1 1]);
            [contourPlot contourPlotVals] = contourf(1E3.*Wpxzs,1E3.*Wpys,nMaxAllSq,15);
            %clabel(contourPlot,contourPlotVals,'LabelSpacing',150); % Put photon number labels on plot
            contourcbar;
            xlabel('W_xz [mm]'); ylabel('W_y [mm]'); grid on;

            % Surface plot for n
            figure('units','normalized','outerposition',[0 0 1 1]);
            surf(1E3.*Wpxzs,1E3.*Wpys,nMaxAllSq);
            xlabel('W_xz [mm]'); ylabel('W_y [mm]'); zlabel('n'); grid on;

            % Contour plot for Ne
            figure('units','normalized','outerposition',[0 0 1 1]);
            [contourPlot2 contourPlotVals2] = contourf(1E3.*Wpxzs,1E3.*Wpys,NeMaxAllSq./N,15);
            %clabel(contourPlot2,contourPlotVals2,'LabelSpacing',150); % Put Ne/N number labels on plot
            %clegendm(contourPlotVals2,contourPlot2);
            contourcbar;
            xlabel('W_xz [mm]'); ylabel('W_y [mm]'); grid on;

            % Surface plot for Ne
            figure('units','normalized','outerposition',[0 0 1 1]);
            surf(1E3.*Wpxzs,1E3.*Wpys,NeMaxAllSq./N);
            xlabel('W_xz [mm]'); ylabel('W_y [mm]'); zlabel('Ne/N'); grid on;
        end
    end
    
    if simPulse == 1
        figure('units','normalized','outerposition',[0 0 1 1]); % Excited atoms
        hold on;
        plot(1E6.*t(1:1:length(t)),100.*pope(1:1:length(pope))./N,'r-','LineWidth',2);
        if oplotNe == 1
            import = importdata(NeFile);
            plot(1E-3.*import(:,1)+tpi.*1E6,import(:,2),'ro');
            ymax = ceil(100.*max([max(real(pope))./N max(import(:,2)./100)]));
            line(1E6.*[tpi tpi], [0 ymax], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
            line(1E6.*[tpi+Tpi tpi+Tpi], [0 ymax], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse        
            legend('Simulation','Data','Pulse start','Pulse end','Location','NorthEast');
        else
            ymax = ceil(100.*max(real(pope)./N));
            line(1E6.*[tpi tpi], [0 ymax], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
            line(1E6.*[tpi+Tpi tpi+Tpi], [0 ymax], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
            legend('Simulation','Pulse start','Pulse end','Location','NorthEast');
        end
        axis([1E6.*tpi-1 ceil(1E6.*(tpi+Tpi))+1 0 ymax]);
        xlabel('Time [탎]'); ylabel('Excitation Rate [%]'); grid on;
        hold off;

        figure('units','normalized','outerposition',[0 0 1 1]); % Atomic coherence
        plot(1E6.*t(1:1:length(t)),coh,'r-');
        line(1E6.*[tpi tpi], [0 max(real(coh))], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 max(real(coh))], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
        axis([1E6.*tpi-1 1E6.*tpi+9 0 max(real(coh))]);
        xlabel('Time [탎]'); ylabel('coh'); grid on;

%         figure('units','normalized','outerposition',[0 0 1 1]); % Cavity photons
%         plot(1E6.*t(1:1:length(t)),n(1:1:length(n)),'r-');
%         line(1E6.*[tpi tpi], [0 max(n(1:1:length(n)))], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
%         line(1E6.*[tpi+Tpi tpi+Tpi], [0 max(n(1:1:length(n)))], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
%         axis([1E6.*tpi-1 1E6.*tpi+9 0 max(n(1:1:length(n)))]);
%         xlabel('Time [탎]'); ylabel('n'); grid on;
        
        figure('units','normalized','outerposition',[0 0 1 1]); % Cavity transmission
        hold on;
        plot(1E6.*t,1E9.*Pout,'r-');
        if oplotP == 1
            import = importdata(Pfile);
            plot(1E6.*(import(:,1)+tpi+tPoffset),import(:,2),'b-');
            legend('Simulation','Data','Location','NorthEast');
        end
        axis([1E6.*tpi-1 1E6.*tpi+9 0 ymaxCT]);
        line(1E6.*[tpi tpi], [0 ymaxCT], 'Color', 'k','LineWidth',2); % Plot line for start of pi-pulse
        line(1E6.*[tpi+Tpi tpi+Tpi], [0 ymaxCT], 'Color', 'k','LineWidth',2); % Plot line for end of pi-pulse
        if simBlue == 1
            line(1E6.*[tblue1 tblue1], [0 ymaxCT], 'Color', 'b','LineWidth',2); % Plot line for blue laser toggle
            line(1E6.*[tblue2 tblue2], [0 ymaxCT], 'Color', 'b','LineWidth',2); % Plot line for blue laser toggle
        end
        xlabel('Time [탎]'); ylabel('P_{out} [nW]'); grid on; hold off;
    end
end

if plotCloudExc == 1
    disp('Rendering excitation animation.');
    bgColCloud = 0.5.*[1 1 1];
    bgColGUI = 1.*[1 1 1];
    xresCloudExc = 1280;
    yresCloudExc = 720;
    figAnim = figure('units','pixels','outerposition',[0 0 xresCloudExc+300 yresCloudExc+300]); % Excitation animation
    axHan=axes('Units','Pixels','Position',[100,100,xresCloudExc,yresCloudExc]); % Plotting area
    ExcitationVid = VideoWriter(['Excitation Animation',datestr(now,' yyyymmdd-HHMMSS'),'.avi']);
    open(ExcitationVid);
    hold(axHan,'on');
    set(gca,'Color',bgColGUI);
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    xlabel(axHan,'x position','Color',[0 0 0]);
    ylabel(axHan,'y position','Color',[0 0 0]);
    yMaxPlot = 1.05.*max(max(r));
    xMaxPlot = 1.4.*yMaxPlot;
    axis(axHan,[-xMaxPlot xMaxPlot -yMaxPlot yMaxPlot]);
    text(-0.97.*xMaxPlot,0.92.*yMaxPlot,'Sr-88 Simulation','FontSize',15,'Color',[0 0 1]);
    if simAOMopen == 0
        text(-0.97.*xMaxPlot,0.56.*yMaxPlot,['Pulse power: ',num2str(round(Ppi.*1E3,1),'%3.0f'),' mW']);
    end
    text(-0.97.*xMaxPlot,0.48.*yMaxPlot,['Pulse waist: (',num2str(round(Wpy*1E3,1),'%3.1f'),' \cdot ',num2str(round(Wpxz*1E3,1)),') mm']);
    text(-0.97.*xMaxPlot,0.40.*yMaxPlot,['Seed power: ',num2str(round(Pseed.*1E9,1),'%3.0f'),' nW']);
    text(-0.97.*xMaxPlot,0.32.*yMaxPlot,['Seed waist: ',num2str(round(W*1E6,0),'%3.0f'),' 탆']);
    text(-0.97.*xMaxPlot,0.24.*yMaxPlot,['Temperature: ',num2str(round(T.*1E3,1),'%3.1f'),' mK']);
    text(-0.97.*xMaxPlot,0.16.*yMaxPlot,['Atoms: ',num2str(round(N./1E6),'%3.0f'),' \cdot 10^6']);
    text(-0.97.*xMaxPlot,0.08.*yMaxPlot,['Atoms per dot: ',num2str(round(Npg))]);
    [excMax,excMaxInd] = max(mean(PeePlot,1));
    text(-0.97.*xMaxPlot,-0.02.*yMaxPlot,['Max excited:',newline,num2str(round(100.*excMax./Npg,1),'%3.1f'), ' % @ t = ',num2str(round((tPlot(excMaxInd)-t(tsteps2)-ntauAOMs.*tauAOM).*1E9,0),'%4.0f'),' ns']);
    text(-0.97.*xMaxPlot,-0.12.*yMaxPlot,['Final excited: ',num2str(round(100.*mean(PeePlot(:,end-5))./Npg,1),'%3.1f'), ' %']);
    cmap = zeros(Ng,3); % Color map
    if Ng < 500
        excPltMarkSize = 50;
    elseif Ng < 2000
        excPltMarkSize = 30;
    elseif Ng < 5000
        excPltMarkSize = 20;
    elseif Ng < 20000
        excPltMarkSize = 10;
    elseif Ng < 100000
        excPltMarkSize = 5;
    elseif Ng < 200000
        excPltMarkSize = 3;
    else
        excPltMarkSize = 2;
    end
        
    for pstep = 1:1:plotSteps-1
        if tPlot(pstep) > 0
            cmap(:,1) = PeePlot(:,pstep)./Npg; % Red color based on excitation
            rectangle('Position',[-1.4*0.99 0.60 0.4 0.24].*yMaxPlot,'EdgeColor',bgColGUI,'FaceColor',bgColGUI);
            rectangle('Position',[-1 -1 2 2].*yMaxPlot./1.03,'EdgeColor',[0 0 0],'FaceColor',bgColCloud,'LineWidth',2);
            line([-0.95.*yMaxPlot -0.95.*yMaxPlot+1E-3],0.92.*[yMaxPlot yMaxPlot],'LineWidth',2,'Color',[0 0 0]);
            if simAOMopen == 1
                rectangle('Position',[-1.4*0.99 0.52 0.4 0.08].*yMaxPlot,'EdgeColor',bgColGUI,'FaceColor',bgColGUI);
                text(-0.97.*xMaxPlot,0.56.*yMaxPlot,['Pulse power: ',num2str(round(PpiStepPlot(pstep).*1E3,1),'%3.0f'),'/',num2str(round(Ppi.*1E3,1),'%3.0f'),' mW']);
            end
            text(-0.95.*yMaxPlot,0.86.*yMaxPlot,'1 mm','color',[0 0 0]);
            if Wpy < 0.8.*yMaxPlot
                line(-0.97.*[yMaxPlot 0.95.*yMaxPlot],[Wpy Wpy],'LineWidth',2,'Color',[1 0 0]);
                line(-0.97.*[yMaxPlot 0.95.*yMaxPlot],[-Wpy -Wpy],'LineWidth',2,'Color',[1 0 0]);
                line(0.97.*[0.95.*yMaxPlot yMaxPlot],[Wpy Wpy],'LineWidth',2,'Color',[1 0 0]);
                line(0.97.*[0.95.*yMaxPlot yMaxPlot],[-Wpy -Wpy],'LineWidth',2,'Color',[1 0 0]);
                text(-0.95.*yMaxPlot,-Wpy-0.04.*yMaxPlot,'2 \cdot Pulse waist','color',[1 0 0]);
            end
            scatter(axHan,squeeze(rPlot(:,1,pstep)),squeeze(rPlot(:,2,pstep)),excPltMarkSize,cmap,'.');
            text(-0.97.*xMaxPlot,0.80.*yMaxPlot,['^{1}S_0 population: ',num2str(round(100.*mean(1-PeePlot(:,pstep)./Npg),1),'%3.1f'),' %'],'color',[0 0 0]);
            text(-0.97.*xMaxPlot,0.72.*yMaxPlot,['^{3}P_1 population: ',num2str(round(100.*mean(PeePlot(:,pstep)./Npg),1),'%3.1f'),' %'],'color',[1 0 0]);
            if simAOMopen == 1
                pulseDurDisp = tPlot(pstep)-t(tsteps2)-ntauAOMs.*tauAOM;
                if pulseDurDisp < 0
                    pulseDurDisp = 0;
                elseif pulseDurDisp > Tpi-2.*ntauAOMs.*tauAOM
                    pulseDurDisp = Tpi-2.*ntauAOMs.*tauAOM;
                end
                text(-0.97.*xMaxPlot,0.64.*yMaxPlot,['Pulse duration: ',num2str(round(pulseDurDisp.*1E9,0),'%4.0f'),' ns']);
            else
                text(-0.97.*xMaxPlot,0.64.*yMaxPlot,['Pulse duration: ',num2str(round((tPlot(pstep)-t(tsteps2)).*1E9,0),'%4.0f'),' ns']);
            end
            frame = getframe(axHan);
            writeVideo(ExcitationVid,frame);
            if pstep == 1
                for l2 = 1:1:14
                    writeVideo(ExcitationVid,frame); % Pause at start for 0.5 seconds
                end
            end
        end
    end
    close(ExcitationVid); close(figAnim);
    disp('Excitation animation completed.');
end

%                 PggPlot = zeros(Ng,plotSteps); % For plotting exc rate
%                 PeePlot = zeros(Ng,plotSteps); % For plotting exc rate
%                 Rplot = zeros(Ng,plotSteps); % For plotting exc rate
%                 tplot = zeros(1,plotsteps);

if simAOMopen == 1
    figure('units','normalized','outerposition',[0 0 1 1]); % Modulated pi pulse power from AOM opening
    hold on;
    set(gca,'FontSize',15);
    plot(1E6.*t(tsteps2:tsteps3-1),PpiStep'.*1e3,'r-','LineWidth',2);
    %axis([0 1E6.*max(t) 0 ymax]);
    xlabel('Time [탎]'); ylabel('Pi pulse power [mW]'); grid on;
    hold off;
end

if saveNeDat == 1 && Tpi < 1 % Save only if Tpi is not above 1s
    %stepsNeDatMax = floor(tsteps1+Tpi./dt);
    stepsNeDatMax = tsteps1+8000;
    tDat = t(tsteps1:stepsNeDatMax)-tpi;
    popeDat = pope(tsteps1:stepsNeDatMax)./N;
    PoutDat = 1E9.*Pout(tsteps1:stepsNeDatMax);
    AA_Ne = [tDat popeDat']; % Output matrix
    if atomRuns > 1
        AA_Pout = [tDat PoutDat']; % Output matrix
    else
        AA_Pout = [tDat PoutDat]; % Output matrix
    end
end