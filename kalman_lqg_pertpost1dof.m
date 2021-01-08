function [XSim_out, USim, sout,Xhat_out,Xstar_out]   = kalman_lqg_pertpost1dof(L,K,Lscale,Kscale,Hscale,H_scale,A_scale,B_scale,Cscale,C0scale,Dscale,D0scale,E0scale,S1scale,FBType,DeficitType,pertsize,posturetime,nTrials,view_flag)
% FBType:
% 'xhat'= Original Todorov's method
%     Stochastic optimal control and estimation methods adapted to the noise characteristics of the sensorimotor system
%     Todorov E (2005). Neural Computation, 17(5): 1084-1108
%     https://homes.cs.washington.edu/~todorov/papers.html
% 'xstar2'= Fred's method
%    Improving the state estimation for optimal control of stochastic processes subject to multiplicative noise. 
%    Crevecoeur, F., Sepulchre, R.J., Thonnard, J.-L., and Lef?vre, P. (2011). Automatica 47, 591?596.


% clear all;
% clear variables;

if(nargin<1)
    L       = [];
    K       = [];
    Lscale  = 1.0;
    Kscale  = 1.0;
    Hscale  = 1.0;
    H_scale = 1.0;
    A_scale = 1.0;
    B_scale = 1.0;
    Cscale      = 0.5;  % xstar2{0.5},xstar{0.5}
    C0scale     = 0.0;  % xstar2{0.0},xstar{0.0}            given as a std
    Dscale      = 0.0;    % xstar2{0.0},xstar{0.0}
    D0scale     = 1e-5; % xstar2{1e-5},xstar{sqrt(10^-10)}  given as a std
    E0scale     = 1e-4; % xstar2{1e-4},xstar{sqrt(10^-6)}   given as a std
    S1scale     = 0.0;  % xstar2{0.0},xstar{0.0}            given as a covariance
    FBType      = 'xstar2';%  % 'xhat' or 'xstar', 'xstar2'
    DeficitType = 'DownScale'; % 'DownScale' or 'NoiseAdd'
    pertsize    = 2;
    posturetime = 1;    % sec
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<3)
    Lscale  = 1.0;
    Kscale  = 1.0;
    Hscale  = 1.0;
    H_scale = 1.0;
    A_scale = 1.0;
    B_scale = 1.0;
    Cscale      = 0.5;  % xstar2{0.5},xstar{0.5}
    C0scale     = 0.0;  % xstar2{0.0},xstar{0.0}            given as a std
    Dscale      = 0.0;    % xstar2{0.0},xstar{0.0}
    D0scale     = 1e-5; % xstar2{1e-5},xstar{sqrt(10^-10)}  given as a std
    E0scale     = 1e-4; % xstar2{1e-4},xstar{sqrt(10^-6)}   given as a std
    S1scale     = 0.0;  % xstar2{0.0},xstar{0.0}            given as a covariance
    FBType      = 'xstar2';  % 'xhat' or 'xstar'
    DeficitType = 'DownScale'; % 'DownScale' or 'NoiseAdd'
    pertsize    = 2;
    posturetime = 3.0;    % sec
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<9)
    Cscale      = 0.5;  % xstar2{0.5},xstar{0.5}
    C0scale     = 0.0;  % xstar2{0.0},xstar{0.0}            given as a std
    Dscale      = 0.0;    % xstar2{0.0},xstar{0.0}
    D0scale     = 1e-5; % xstar2{1e-5},xstar{sqrt(10^-10)}  given as a std
    E0scale     = 1e-4; % xstar2{1e-4},xstar{sqrt(10^-6)}   given as a std
    S1scale     = 0.0;  % xstar2{0.0},xstar{0.0}            given as a covariance
    FBType      = 'xstar2';  % 'xhat' or 'xstar'
    DeficitType = 'DownScale'; % 'DownScale' or 'NoiseAdd'
    pertsize    = 2;
    posturetime = 3.0;    % sec
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<15)
    FBType      = 'xstar2';  % 'xhat' or 'xstar'
    DeficitType = 'DownScale'; % 'DownScale' or 'NoiseAdd'
    pertsize    = 2;
    posturetime = 3.0;    % sec
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<16)
    DeficitType = 'DownScale'; % 'DownScale' or 'NoiseAdd'
    pertsize    = 2;
    posturetime = 3.0;    % sec
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<17)
    pertsize    = 2;
    posturetime = 3.0;    % sec
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<18)
    posturetime = 3.0;    % sec
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<19)
    nTrials     = 30;
    view_flag   = 1;
elseif(nargin<20)
    view_flag   = 1;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define parameters


prm.dt = 0.01;        % time step (sec)
prm.m = 1;            % mass (kg)
prm.b = 1;            % damping (N/sec)
prm.tau = 40;         % time constant (msec)

prestab     = 0.5;    % Time stable at the end target (sec)
movetime    = 0.5;    % Movement duration (sec)
% poststab    = 3.0;    % Time stable at the end target (sec)
poststab    = posturetime;    % Time stable at the end target (sec)
delay       = .05;    % Sensorimotor delay (sec)
% delay       = 0;    % Sensorimotor delay (sec)


prm.C   = Cscale;%0.4;        % control-dependent noise
prm.C0  = [0 0 1 0]*C0scale;%[0 0 1 0 0 0]*C0scale;%10^-6;    % motor noise [pos vel force extforce]
prm.D   = Dscale;%0;%[1 1 1 1]*0.2;%0;          % state-dependent noise
prm.D0  = [1 1 1 1]*D0scale;%[1 1 1 1 1 1]*D0scale;%10^-8;%[0.02, 0.2, 1.0, 1.0]*0.5;    % sensory noise [pos vel force extforce]
prm.E0  = [1 1 1 1]*E0scale;%[1 1 1 1 1 1]*E0scale;%10^-6;   % internal noise [pos vel force extforce]

prm.H   = [1 1 1 1];%[1 1 1 1 1 1];

prm.r = 10^-6;%((prestab+movetime+poststab)/delay)*10^-6;%0.00001;     % control signal penalty
prm.q = [1 1 0 0];%[1 0 0 0 -1 0;0 1 0 0 0 -1];%[0.2, 0.02, 0, 0];     % error penalty [pos vel force extforce]

X1  = [0 0 0 0]';%[0 0 0 0 0 0]';
S1  = [1 1 1 1]*S1scale;%10^-6;

% sim
% nTrials        = 30;     % Number of simulated trajectories
% pertsize    = 2;      % N
% FBType      = 'xstar';    % 'xhat' or 'xstar'
% FBType      = 'xhat';

% Lscale  = 1.0;
% Kscale  = 1.0;
% Hscale  = 1.0;
% H_scale = 1.0;
% A_scale = 1.0;
% B_scale = 1.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute system dynamics and cost matrices

dtt = prm.dt/(prm.tau/1000);

% compute times
prestab_npt     = round(prestab/prm.dt);
movetime_npt    = round(movetime/prm.dt);
poststab_npt    = round(poststab/prm.dt);
stab_flag       = [true(1,prestab_npt+1),false(1,movetime_npt),true(1,poststab_npt)];
N               = 1+prestab_npt+movetime_npt+poststab_npt;
nDelay          = round(delay/prm.dt); % sonsorymotor delay in timepoints
T               = ((1:N) - 1 - prestab_npt) * prm.dt;
P               = [zeros(1,prestab_npt+1),ones(1,movetime_npt),ones(1,poststab_npt)]*pertsize;

szX = size(X1,1);
szU = 1;
szY = size(prm.H,2);

A = eye(szX,szX);
A(1,1) = 1;
A(1,2) = prm.dt;
A(2,2) = 1-prm.dt*prm.b/prm.m;
A(2,3) = prm.dt/prm.m;
A(2,4) = prm.dt/prm.m;
A(3,3) = 1-dtt;
A(4,4) = 1;

B = zeros(szX,szU);
B(3,1) = dtt;

C   = prm.C;      % variance of control-dependent-noise
C0  = diag(prm.C0);   % std of additive motor noise

H   = zeros(szY,szX);
H(1:szY,1:szY)  = diag(prm.H);

D   = diag(prm.D);      % weight of state-dependent-noise
D0  = diag(prm.D0); % diag([prm.pos prm.vel prm.frc prm.exfrc 0]); % std of additive sensory noise

E0  = diag(prm.E0); % 0.5;     % std of internal noise

R   = prm.r;  % weight for motor cost

Q   = zeros(szX,szX,N);       % werights for error cost
if(dimens(prm.q)==1)
    d   = diag(prm.q);
else
    d   =prm.q;
end
Q(:,:,stab_flag)    = repmat(d'*d,[1,1,sum(stab_flag)]);

% X1 = zeros(szX,1);
S1 = diag(S1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Augument parameters with sensory motor delay
szX = size(A,1);
szU = size(B,2);
szY = size(H,1);

A   = [A,zeros(szX,szX*nDelay)];
A   = [A;[eye(szX*nDelay),zeros(szX*nDelay,szX)]];
B   = [B;zeros(szX*nDelay,szU)];

H   = [zeros(szY,szX*nDelay),H];
% H   = [zeros(szX*nDelay,szX*(nDelay+1));H];

if(length(C0(:))>1)
    C0  = [C0;zeros(szX*nDelay,size(C0,2))];
end

if(length(D0(:))>1)
%     D0  = [zeros(szX,szX*nDelay),D0];
%     D0  = [zeros(szX*nDelay,szX*(nDelay+1));D0];
    D0  = D0; % because y is not affected by time-delay
end
% if(length(D(:))>1)
%     D   = [zeros(szX,szX*nDelay),D];
%     D   = [zeros(szX*nDelay,szX*(nDelay+1));D];
% end

if(length(E0(:))>1)
    E0  = [E0,zeros(szX,szX*nDelay)];
    E0  = [E0;zeros(szX*nDelay,szX*(nDelay+1))];
end

Q   = cat(2,Q,zeros(szX,(szX*nDelay),N));
Q   = cat(1,Q,zeros(szX*nDelay,szX*(nDelay+1),N));

X1  = [X1;zeros(szX*nDelay,1)];
S1  = diag(repmat(diag(S1),[nDelay+1,1]));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optimize K and L
if(isempty(L) || isempty(K))
    switch FBType
        case {'xhat','xstar'}
            [K,L,Cost] = kalman_lqg_xhat(A,B,C,C0,H,D,D0,E0,Q,R,X1,S1);
        case 'xstar2'
            [K,L,Cost] = kalman_lqg_xstar(A,B,C,C0,H,D,D0,E0,Q,R,X1,S1);
    end
else
    Cost    = [];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation
% Copy Internal model
H_  = H;
A_  = A;
B_  = B;

switch DeficitType
    case 'DownScale' % Deactivation
        L   = Lscale*L;
        K   = Kscale*K;
        H   = Hscale*H;
        A_  = A_scale*A_;
        B_  = B_scale*B_;
        H_  = H_scale*H_;
    case 'NoiseAdd' % NoiseAddition
        L   = L + abs(L) .* Lscale .* randn(size(L));
        K   = K + abs(K) .* Kscale .* randn(size(K));
        H   = H + abs(H) .* Hscale .* randn(size(H));
        A_  = A_+ abs(A_).* A_scale.* randn(size(A_));
        B_  = B_+ abs(B_).* B_scale.* randn(size(B_));
        H_  = H_+ abs(H_).* H_scale.* randn(size(H_));
end

[XSim,USim,CostSim,Xhat,Xstar] = ...
   kalman_lqg_pertpost1dof_sim(L,K,A,B,H,A_,B_,H_,C,C0,D,D0,E0,Q,R,X1,S1,nTrials,P,FBType);


XSim_out    = XSim(1:szX,:,:);
Xhat_out    = Xhat(1:szX,:,:);
Xstar_out   = Xstar(1:szX,:,:);

sout.L      = L;
sout.K      = K;
sout.CostSim    = CostSim;
sout.Cost   = Cost;
sout.time   = T;
sout.P      = P;
sout.A      = A;
sout.B      = B;
sout.H      = H;
sout.C      = C;
sout.C0     = C0;
sout.D      = D;
sout.D0     = D0;
sout.E0     = E0;
sout.R      = R;
sout.Q      = Q;
sout.Lscale = Lscale;
sout.Kscale = Kscale;
sout.Hscale = Hscale;
sout.H_scale    = H_scale;
sout.A_scale    = A_scale;
sout.B_scale    = B_scale;
sout.Cscale     = Cscale;
sout.C0scale    = C0scale;
sout.Dscale     = Dscale;
sout.D0scale    = D0scale;
sout.E0scale    = E0scale;
sout.S1scale    = S1scale;
sout.FBType     = FBType;  % 'xhat' or 'xstar'


if(view_flag== 1)
    figure(100)
    plot(T,squeeze(XSim(1,:,:)));ylim([-0.01 0.03]);xlim([-0.5 1]);set(gca,'YGrid','on','XGrid','on')
    K(:,:,100)
end


% clf;
% plot(squeeze(XSim(1,:,:))','r'); hold on;
% plot(Xa(1,:),'k','linewidth',2);
% xlabel('time step');
% ylabel('position');

end