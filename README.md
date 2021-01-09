# FreeCodeCamp Projects

This is my repo for holding all fcc projects, so far from the Machine Learning and Data Visualization modules with [d3.js](https://d3js.org). 

## Dependencies for Machine Learning
* Tensorflow >= 2.x
* Numpy 
* Matplotlib

All dependencies can be be installed with pip or the code can be directly from the Google Colab link. 
Colab is a cloud hosted version of IPython Jupyter notebooks used for development. 

```bash
pip install tensorflow 
pip install matplotlib
pip install numpy
```
## Running the D3 Projects
The d3.js projects were compiled with [Babel](https://babeljs.io), to run the tests you have to include Babel:

```bash
npm install --save-dev @babel/core @babel/cli
```
Make sure to include d3.js in local html files if you want to run your own HTML
```html
<script src="https://d3js.org/d3.v6.min.js"></script>
```
