# Simple Linear regression
### Approach
1. We need total 4 function to implement SLR. That is .fit(), .predict(), .cost(). .score()
2. For fir function we need values of X and y and then since cost should me minimum. Therefore differentiating it with respect to m and c will give formula of m and c.
3. Therefore via that formula we get m and c. Now we return them
4. Then in predict function we use x, m, c to find y_pred then return it
5. Then in score function we calculate the score(r**2) which is given by 1-(u/v)
6. Then at last cost is give by MSE(mean squared error)

### This approach is good but it is not properly optimized
#### X and y must be of same length
#### length of X and y must not be 0
#### X should contain more than 1 unique value as if not then it will generate the 0 division error while calculating value of m in fit function. As while calculating m the denominator will become 0

------------------------------------------
SLR is failing for 

X = [5, 5, 5, 5]  
y = [30, 35, 40, 45]
------------------------------------------
#### Therefore Enhanced SLR came into picture
### Approach for EnhancedSLR
1. Added check for length of x and y
2. Added check that x must have more than 1 unique value