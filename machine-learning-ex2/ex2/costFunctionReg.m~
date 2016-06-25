function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


for i =1:m

	if y(i)==0
		J=J-log(1-sigmoid(X(i,:)*theta));
	end
	if y(i)==1
		J=J-log(sigmoid(X(i,:)*theta));

	end
end
J=J+ (lambda/2)*((transpose(theta)*theta)-theta(1)^2)
J=J/m;

p = (sigmoid(X*(theta))-y) ;
grad(1) = transpose(X(:,1))*p;

for j= 2: size(theta)
	
	grad(j)= transpose(X(:,j))*p + lambda*theta(j);
end

grad = grad/m;
	

% =============================================================

end
