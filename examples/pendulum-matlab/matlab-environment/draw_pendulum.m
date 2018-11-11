%% draw_pendulum.m
% *Summary:* Draw the pendulum system with reward, applied torque, 
% and predictive uncertainty of the tips of the pendulums
%
%    function draw_pendulum(theta, torque, cost, text1, text2, M, S)
%
%
% *Input arguments:*
%
%   theta1     angle of inner pendulum
%   theta2     angle of outer pendulum
%   f1         torque applied to inner pendulum
%   f2         torque applied to outer pendulum
%   cost       cost structure
%     .fcn     function handle (it is assumed to use saturating cost)
%     .<>      other fields that are passed to cost
%   text1      (optional) text field 1
%   text2      (optional) text field 2
%   M          (optional) mean of state
%   S          (optional) covariance of state
%
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-18


function draw_pendulum(theta, torque, cost, text1, text2, M, S)
%% Code

l = 0.6;
xmin = -1.2*l; 
xmax = 1.2*l;    
umax = 0.5;
height = 0;

% Draw pendulum
pendulum = [0, 0; l*sin(theta), -l*cos(theta)];
clf; hold on
plot(pendulum(:,1), pendulum(:,2),'r','linewidth',4)

% plot ellipses around tips of pendulum (if M, S exist)
try
  if max(max(S))>0
    err = linspace(-1,1,100)*sqrt(S(2,2));
    plot(l*sin(M(2)+2*err),-l*cos(M(2)+2*err),'b','linewidth',1)
    plot(l*sin(M(2)+err),-l*cos(M(2)+err),'b','linewidth',2)
    plot(l*sin(M(2)),-l*cos(M(2)),'b.','markersize',20)
  end
catch
end

% Draw useful information
% target location
plot(0,l,'k+','MarkerSize',20);
plot([xmin, xmax], [-height, -height],'k','linewidth',2)
% joint
plot(0,0,'k.','markersize',24)
plot(0,0,'y.','markersize',14)
% tip of pendulum
plot(l*sin(theta),-l*cos(theta),'k.','markersize',24)
plot(l*sin(theta),-l*cos(theta),'y.','markersize',14)
plot(0,-2*l,'.w','markersize',0.005)
% applied torque
plot([0 torque/umax*xmax],[-0.5, -0.5],'g','linewidth',10);
% immediate reward
reward = 1-cost.fcn(cost,[0, theta]',zeros(2));
plot([0 reward*xmax],[-0.7, -0.7],'y', 'linewidth',10);
text(0,-0.5,'applied  torque')
text(0,-0.7,'immediate reward')
if exist('text1','var')  
  text(0,-0.9, text1)
end
if exist('text2','var')  
  text(0,-1.1, text2)
end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-2*l 2*l]);
axis off;
drawnow;