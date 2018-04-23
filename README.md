RBM_Grid_Generator

## PROBLEM DESCRIPTION

The data we want to analyze is styles from a group of elements. In this case we had 4 styles so we want at least one machine for every one that it will be able to analyze the data an generate new data.

So the data that we have is a set of vectors that's the output of the app that generate grids. This grids ara a composition of elements so the outuput will be a vector of id. Each grid had 48 elements and each element can have a maximum of 255 elements in it. But is not necessary to had that number, it can be possible to have less. 

<h4 align="center">$[ x_1, x_2, x_3, ..., x_{48} ]$</h4> 
<h4 align="center">$x_1 = [id_1,id_2,_id_3, ...,id_{255}]$</h4>

Each element of the vector can have a different size from the others and the list of integers are the subcategories of style of the object that was placed in the grid. Each number is a different subcategory of style of the object placed previously.

So the first objective is to descompose the input data, a vector which element is a vector of different size, to a list of vectors that will have de same length, 48, and one element in each position. So the main goal is to find the grid wich elements has more IDs so we can set the size of total vectors, $y$ that we need to descompose the data. In case that one element of the vector has lower IDs that the max we gonna put a $0$ as the null value for our data.

<h4 align="center">$input = [[id_1^1,id_2^1,_id_3^1, ...,id_{255}^1], [id_1^2,id_2^2,_id_3^2, ...,id_{255}^2], ..., [id_1^48,id_2^48,_id_3^48, ...,id_{255}^48]]$</h4> 
<h4 align="center">$output = [[id_1^1, id_1^2,..., id_1^{48}],
[id_2^1, id_2^2,..., id_2^{48}],..., [0,0,...,id_y^{48}]$</h4> 


First of all we gonna implement a function that allows us to obtain the max value of the input data.