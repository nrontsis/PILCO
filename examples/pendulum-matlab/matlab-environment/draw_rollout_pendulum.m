%% draw_rollout_pendulum.m
% *Summary:* Script to draw a trajectory of the most recent pendulum trajectory
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-27
%
%% High-Level Steps
% # For each time step, plot the observed trajectory

%% Code

% Loop over states in trajectory
for r = 1:size(xx,1)
  cost.t = r;
  if exist('j','var') && ~isempty(M{j})
    draw_pendulum(latent{j}(r,2), latent{j}(r,end), cost,  ...
      ['trial # ' num2str(j+J) ', T=' num2str(H*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*size(x,1)) ...
      ' sec'], M{j}(:,r), Sigma{j}(:,:,r));
  else
     draw_pendulum(latent{jj}(r,2), latent{jj}(r,end), cost,  ...
      ['(random) trial # ' num2str(1) ', T=' num2str(H*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*size(x,1)) ...
      ' sec'])
  end
  pause(dt);
end