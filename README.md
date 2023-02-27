# On the Specialization of Neural Modules
Code repository for reproducing the epxeriments from the paper: "On the Specialization of Neural Modules"

To reproduce the linear network results from the paper run: "source run.sh". The version of python used is: Python 3.7.3.
A requirements.txt file is included with the packages and versions needed to run the code.
To install the necessary packages use: pip install -r requirements.txt. Anaconda (4.8.2) was used on our system to manage the install
environment.

The code uses Jax, however, the package installed from the requirements file is the cpu-only version of Jax. The CPU version is however
still sufficient to generate all linear network results and should complete within a minute.
If necessary, more information on installing GPU Jax can be found at: https://github.com/google/jax#installation.
CUDA101 was used for our results.

Many of the plots use latex in the titles. Thus, some additional latex packages are needed which can be installed with:
sudo apt-get install cm-super dvipng texlive-latex-extra texlive-fonts-recommended.

Once "source run.sh" has been run, the directories: dense_network, split_network, shallow_network and other_rules will contain
the desired results corresponding to those found in the paper. Running: "source clean_dir.sh" will remove the generated results if needed and restore the directories
to their original contents.

Comments are left in the code to help explain each part and places where hyper-parameters are set for either the dataset or training
procedure are indicated.

Reproducing the CMNIST results will take longer and it is recommended that the GPU version of Jax be used (results take roughly 4 hours with GPU).
For convenience a second bash script is used to reproduce the CMNIST results and can be run using: source run_cmnist.sh. The desired results
will be inside the cmnist directory. In the paper the CMNIST results are averaged over 10 run but we have left the code here using 3 runs. 
Run: "source clean_cmnist.sh" to restore the cmnist directory to its original state.

Finally, to reproduce the motivating example in Appendix A run: "source run_motivation.sh". To clean the directory run: "source clean_motivation.sh"
