function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

%% set x_1: exam1 marks and x_2: exam2 marks; 
%% x_1 and x_2 as X,Y axis of plots
X_1 = X(:,1);
X_2 = X(:,2);
idx = y==1;
plot(X_1(idx),X_2(idx),'k+');
plot(X_1(~idx),X_2(~idx),'ko');

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================



hold off;

end
