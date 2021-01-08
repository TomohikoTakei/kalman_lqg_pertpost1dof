%[K,L,Cost,Xa,XSim,CostSim] = 
%   kalman_lqg_sim( A,B,C,C0, H,D,D0, E0, Q,R, X1,S1  [NSim,Init,Niter] )
%
% Simulate noisy trajectories based on parameters from Kalman_lqg
%


function [XSim,USim,CostSim,Xhat,Xstar] = ...
   kalman_lqg_pertpost1dof_sim(L,K,A,B,H,Ahat,Bhat,Hhat,C,C0,D,D0,E0,Q,R,X1,S1,NSim,P,FBType)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization

% determine sizes
szX = size(A,1);
szU = size(B,2);
szY = size(H,1);
szC = size(C,3);
szC0 = size(C0,2);
szD = size(D,3);
szD0 = size(D0,2);
szE0 = size(E0,2);

N   = size(L,3)+1;

% if C or D are scalar, replicate them into vectors
if size(C,1)==1 & szU>1,
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
if length(C0(:))==1 & C0(1)==0,
    C0 = zeros(szX,1);
end
if length(D0(:))==1 & D0(1)==0,
    D0 = zeros(szY,1);
end
if length(E0(:))==1 & E0(1)==0,
    E0 = zeros(szX,1);
end

% for Fred algorithm
if size(C,1)==1
    C   = C.*B;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulate noisy trajectories

% square root of S1
[u,s,v] = svd(S1);
sqrtS = u*diag(sqrt(diag(s)))*v';

% initialize
XSim = zeros(szX,NSim,N);
Xhat = zeros(szX,NSim,N);
Xstar= zeros(szX,NSim,N);
USim = zeros(1,NSim,N);
Xhat(:,:,1) = repmat(X1, [1 NSim]);
XSim(:,:,1) = repmat(X1, [1 NSim]) + sqrtS*randn(szX,NSim);

CostSim = 0;

% loop over N
for k=1:N-1
    
    % update control and cost
    U = -L(:,:,k)*Xhat(:,:,k);
    USim(:,:,k) = U;
    CostSim = CostSim + sum(sum(U.*(R*U)));
    CostSim = CostSim + sum(sum(XSim(:,:,k).*(Q(:,:,k)*XSim(:,:,k))));
    
    % compute sdn
    sdn = 0;
    for i=1:szC
        sdn = sdn + (C(:,:,i)*U).*repmat(randn(1,NSim),[szX 1]);
    end;
    
    % compute real next state
    if(~isempty(P) && length(P)==N)
        XSim(4,:,k) = P(k);     % Add external perturbation
    end
    XSim(:,:,k+1)   = A*XSim(:,:,k) + B*U + C0*randn(szC0,NSim) + sdn; % compute nextstate
    
    % Predict next state
    prior           = Ahat*Xhat(:,:,k) + Bhat*U  + E0*randn(szE0,NSim);  % predict next state
    Xstar(:,:,k+1)  = prior;
    
    
    switch lower(FBType)
        case 'xhat' % Todorov 2005
            % compute current statedn
            statedn = 0;
            for i = 1:szD
                statedn = statedn + (D(:,:,i)*XSim(:,:,k)).*repmat(randn(1,NSim),[szY 1]);
            end
            
            % compute current observation
            y = H*XSim(:,:,k) + D0*randn(szD0,NSim) + statedn;
            
            % Estimate next state
            Xhat(:,:,k+1) = prior + K(:,:,k)*(y-Hhat*Xhat(:,:,k));
            
        case {'xstar','xstar2'}    % Fred 2011
            % compute next statedn
            statedn = 0;
            for i = 1:szD
                statedn = statedn + (D(:,:,i)*XSim(:,:,k+1)).*repmat(randn(1,NSim),[szY 1]);
            end
            
            % compute next observation
            y = H*XSim(:,:,k+1) + D0*randn(szD0,NSim) + statedn;
            
            % Estimate next state
            Xhat(:,:,k+1) = prior + K(:,:,k)*(y-Hhat*prior);
    end
end;

% final cost update
CostSim = CostSim + sum(sum(XSim(:,:,N).*(Q(:,:,N)*XSim(:,:,N))));
CostSim = CostSim / NSim;
