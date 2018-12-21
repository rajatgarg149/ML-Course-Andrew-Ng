function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Z1 = [ones(m, 1) X] * Theta1';
A1 = sigmoid(Z1);
Z2 = [ones(m, 1) A1] * Theta2';
A2 = sigmoid(Z2);

K = size(A2, 2);            %Number of labels
Y = zeros(size(y, 1), K);
for i = 1:K,
  Y(:, i) = (y == i);
end;

J = - sum(sum(Y.*log(A2) + (1-Y).*log(1-A2))) / m;
Theta2_ntbias = Theta2(:, 2:end);
Theta1_ntbias = Theta1(:, 2:end);
regularized_term = (sum(sum((Theta2_ntbias .* Theta2_ntbias))) + sum(sum(Theta1_ntbias .* Theta1_ntbias))) / m;
J = J + (lambda / 2) * regularized_term;

dA2 = - ((Y ./ A2) - ((1 - Y) ./ (1 - A2)));
dZ2 = dA2 .* sigmoidGradient(Z2);
Theta2_grad = (dZ2' * [ones(m, 1) A1]) / m;
dA1 = dZ2 * Theta2_ntbias;
dZ1 = dA1 .* sigmoidGradient(Z1);
Theta1_grad = (dZ1' * [ones(m, 1) X]) / m;

Theta2_grad_ntbias = Theta2_grad(:, 2:end);
Theta1_grad_ntbias = Theta1_grad(:, 2:end);
regularized_grad2 = Theta2_grad_ntbias + (lambda * Theta2_ntbias) / m;
regularized_grad1 = Theta1_grad_ntbias + (lambda * Theta1_ntbias) / m;

Theta2_grad = [Theta2_grad(:, 1) regularized_grad2];
Theta1_grad = [Theta1_grad(:, 1) regularized_grad1];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
