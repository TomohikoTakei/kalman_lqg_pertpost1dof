%[K,L,Cost,XSim,USim,CostSim] =
%   kalman_lqg_xstar( A,B,C,C0, H,D,D0, E0, Q,R, X1,S1)
%
% Compute optimal controller and estimator for generalized LQG
%
% u(t)    = -L(t) x(t)
% x(t+1)  = A x(t) + B (I + Sum(C(i) rnd_1)) u(t) + C0 rnd_n
% y(t)    = H x(t) + Sum(D(i) rnd_1) x(t) + D0 rnd_n
% xhat(t+1) = A xhat(t) + B u(t) + K(t) (y(t) - H xhat(t)) + E0 rnd_n
% x(1)    ~ mean X1, covariance S1
%
% cost(t) = u(t)' R u(t) + x(t)' Q(t) x(t)

% K       Filter gains
% L       Control gains
% Cost    Expected cost (per iteration)
% XSim    Simulated trajectories
% USim    Simulated motor command  
% CostSim Empirical cost
%
% The basic script is based on the following paper and code is available at:
%  https://homes.cs.washington.edu/~todorov/software/gLQG.zip
%  Todorov, E. (2005) Stochastic optimal control and estimation
%  methods adapted to the noise characteristics of the
%  sensorimotor system. Neural Computation 17(5): 1084-1108

% The code was modified based on Crevecouer 2011:
% Improving the state estimation for optimal control of stochastic processes subject to multiplicative noise. 
%    Crevecoeur, F., Sepulchre, R.J., Thonnard, J.-L., and Lef?vre, P. (2011). Automatica 47, 591?596.

% Copyright (C) Tomohiko Takei, 2022



function [K,L,Cost] = kalman_lqg_xstar( A,B,C,C0, H,D,D0, E0, Q,R, X1,S1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization

% numerical parameters
% MaxIter = 500;
% Eps = 10^-15;
MaxIter = 100;
Eps = 10^-14;

% determine sizes
szX = size(A,1);
szU = size(B,2);
szY = size(H,1);
szC = size(C,3);
szC0 = size(C0,2);
szD = size(D,3);
szD0 = size(D0,2);
szE0 = size(E0,2);
N = size(Q,3);


% if C or D are scalar, replicate them into vectors
if size(C,1)==1 && szU>1,
    C = C*ones(szU,1);
end;
if length(D(:))==1,
    if D(1)==0,
        D = zeros(szY,szX);
    else
        D = D*ones(szX,1);
        if szX ~= szY,
            error('D can only be a scalar when szX = szY');
        end
    end
end;

% if C0,D0,E0 are scalar, set them to 0 matrices and adjust size
if length(C0(:))==1 && C0(1)==0,
    C0 = zeros(szX,1);
end
if length(D0(:))==1 && D0(1)==0,
    D0 = zeros(szY,1);
end
if length(E0(:))==1 && E0(1)==0,
    E0 = zeros(szX,1);
end

% for Fred algorithm
if size(C,1)==1
    C   = C.*B;
end;

% Prepare covariance matrix
oXi     = C0*C0';
oEta    = E0*E0';
oOmega    = D0*D0';

% initialize policy and filter
K = zeros(szX,szY,N-1);
L = zeros(szU,szX,N-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run iterative algorithm - until convergence or MaxIter
for iter = 1:MaxIter
    %% initialize optimal cost-to-go function
    Sx  = Q(:,:,N);
    Se  = zeros(szX,szX);
    Cost(iter)  = 0;
    s   = 0;
    
    % backward pass - recompute control policy
    for k=N-1:-1:1
        
        % Prepare parameters 
        UU = Sx;
        KSeK = K(:,:,k)'*Se*K(:,:,k);
        for i=1:szD
            UU = UU + D(:,:,i)'*KSeK*D(:,:,i);
        end;
        
        delta = eye(szX,szX) - K(:,:,k)*H;
        UdSed = UU + delta'*Se*delta; 
        
        GG   = 0;
        for i=1:szC
            GG = GG + C(:,:,i)'*UdSed*C(:,:,i);
        end;
        
        % Controller
        L(:,:,k) = pinv(B'*UU*B + R + GG)*(B'*UU*A);
        
        % update Sx, Se, and s
        s   = s + trace(UU*oXi) + trace(Se*(delta*(oXi + oEta)*delta' + K(:,:,k)*oOmega*K(:,:,k)'));
        Sx  = Q(:,:,k) + A'*UU*(A - B*L(:,:,k));
        Se  = A'*UU*B*L(:,:,k) + A'*delta'*Se*delta*A;
        
    end;
    
    %% adjust cost
    Cost(iter)      = X1'*Sx*X1 + trace((Se+Sx)*S1) + s;
    
    
    %% initialize covariances
    SiP     = nan(szX,szX,N);
    SiE     = nan(szX,szX,N);
    SiX     = nan(szX,szX,N);
    SiXhat  = nan(szX,szX,N);
    SiXE    = nan(szX,szX,N);
    V       = nan(szX,szX,N);
    W       = nan(szY,szY,N);
    
    SiE(:,:,1)     = S1;
    SiX(:,:,1)     = X1*X1';
    SiXhat(:,:,1)  = X1*X1';
    SiXE(:,:,1)    = zeros(szX,szX);
    
    
    
    % forward pass - recompute Kalman filter
    for k = 1:N-1
        % Prepare parameters
        V(:,:,k) = zeros(szX,szX);
        for i=1:size(C,3)
            V(:,:,k) = V(:,:,k) + C(:,:,i)*L(:,:,k)*SiXhat(:,:,k)*L(:,:,k)'*C(:,:,i)';
        end;
        
        temp = A * SiX(:,:,k) * A' + oXi;
        M = zeros(szY,szY);
        for i=1:szD
            M = M + D(:,:,i)*temp*D(:,:,i)';
        end;
        
        % update SiP;
        SiP(:,:,k+1)  = A*SiE(:,:,k)*A' + oXi + oEta;
        
        % compute Kalman gain
        K(:,:,k) = SiP(:,:,k+1)*H'*pinv(H*SiP(:,:,k+1)*H' + oOmega + M);
        
        % update SiX, SiE, SiXE
        SiX(:,:,k+1)    = A*SiX(:,:,k)*A'  + B*L(:,:,k)*SiXhat(:,:,k)*L(:,:,k)'*B' + oXi + V(:,:,k) - ...
            (A*(SiXhat(:,:,k) + SiXE(:,:,k)')*L(:,:,k)'*B' + B*L(:,:,k)*(SiXhat(:,:,k) + SiXE(:,:,k))*A');
        
        delta = eye(szX,szX) - K(:,:,k)*H;
            W(:,:,k+1) = zeros(szY,szY);
            for i=1:szD
                W(:,:,k+1) = W(:,:,k+1) + D(:,:,i)*SiX(:,:,k+1)*D(:,:,i)';
            end;
        
        SiE(:,:,k+1)     = delta*(A*SiE(:,:,k)*A' + oXi + oEta +V(:,:,k))*delta' +...
            K(:,:,k)*(oOmega + W(:,:,k+1))*K(:,:,k)';
        SiXhat(:,:,k+1)  = (A - B*L(:,:,k))*SiXhat(:,:,k)*(A - B*L(:,:,k))' + delta*oEta*delta' + ...
            delta*(A - B*L(:,:,k))*SiXE(:,:,k)'*A'*H'*K(:,:,k)' + ...
            K(:,:,k)*H*A*SiXE(:,:,k)*(A - B*L(:,:,k))'*delta' + ...
            K(:,:,k)*(H*SiX(:,:,k+1)*H' + oOmega + W(:,:,k+1))*K(:,:,k)';
        SiXE(:,:,k+1)    = ((A - B*L(:,:,k))*SiXE(:,:,k)*A' + K(:,:,k)*H*A*SiE(:,:,k)*A')*delta' - ...
            delta*oEta*delta' - K(:,:,k)*(oOmega + W(:,:,k+1))*K(:,:,k)';
    end;
    
    %% progress bar
    if ~rem(iter,10),
        fprintf('.');
    end;
    
    % check convergence of Cost
    if (iter>1 && abs(Cost(iter-1)-Cost(iter))<Eps)
        break;
    end;
end;

% print result
if Cost(iter-1)~=Cost(iter)
    fprintf(' Log10DeltaCost = %.2f\n',log10(abs(Cost(iter-1)-Cost(iter))));
else
    fprintf(' DeltaCost = 0\n' );
end;
