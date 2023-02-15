<h1>Prediction-as-a-Service Model Stealing</h1>


<h2>Description</h2>

This code demonstrates two ways to steal a linear regression model from SecureHealth’s Prediction-as-a-Service API. 
The code uses Paillier encryption to extract the model parameters via two methods: weight-by-weight extraction and solving a linear equation. In the weight-by-weight approach, the program queries the API by providing a vector of zeros and then sends unitary vectors one by one to extract each weight. In the linear equation approach, the program generates 11 random vectors and solves a linear equation to extract all 11 model parameters.


<h2>Languages Used</h2>

- <b>Python </b> 

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
