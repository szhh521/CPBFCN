Software Requests
Python 3.6
PyTorch 1.1.0
h5py 3.1.0
meme 5.1.1
weblogo 3.7.8
Scipy 1.5.4
torchsummary 1.5.1
scikit-learn 0.24.2

Data process
CircRNA_hg19 URL: http://www.circbase.org/cgi-bin/downloads.cgi
CircRNA-protein binding datasets URL: https://circinteractome.nia.nih.gov/rna_binding_protein.html
python data_process.py -d 'your experimental datasets path' -l  'seq length' -p 'your circRNA_hg19 fasta file path'

Model parameter description
#-d datapath: experimental datasets path
#-l seq_length:  the record length in experimental datasets
#-p hg19_path: CircRNA_hg19 fasta  path

Model Train
python parameter_run_motif.py

Model parameter description
#Instead of setting file parameters, we set a data list for different parameters, and complete parameter transfer and model training by for loop and calling the main model training file.
#data list: seq_len, loss_threshold, loss_function, model_select

Get motif
python CPBFCN_motif.py

Model parameter description
#In this step, you should set five parameters in CPBFCN_motif.py file: trained model path, data path, seq length, motif length, motif result path.
#dpath: trained model path
#npz_path: experimental data path
#seq_len: seq length
#motiflen: motif length
#store_path: directory path for store motif result

Motif distribution figure
##motif distribution in experimental datasets with seqence length 201##
python 201_seq_dist_fig.py

Model parameter description
#in this step, you should first confirm motif seq and its number involved in creating motif distribution figure.
#fig_path: the directory path that store your motif distribution figure.

##motif distribution in RBP dataset or CircRNA##
python motif_dist_rbp_circRNA.py -t 'RBP or CRNA' -r ''G:\CPBPFCN-Result\\all_data\\101\TIA1\positive' -c 'G:\CPBPFCN-Result\data\human_hg19_circRNAs_putative_spliced_sequence.fa' -n 'hsa_circ_0000001' -cm 'UUUUUCUUUU' -mm 'UUUUUUC'

Model parameter description
#-t data_type: input data type (RBP or CircRNA)
#-r experimental datasets:  if the data type is RBP, this path would be used.
#-c circRNA_hg19_path: CircRNA_hg19 fasta file  path
#-n circRNA name: if the data type is CRNA, this parm would be used.
#-cm motif seq found by CPBFCN
#-mm motif seq in MEME database

Acknowledgments
Qinhu Zhang, Siguo Wang, Zhanheng Chen, Ying He, Qi Liu and De-Shuang Huang. Locating transcription factor binding sites by fully convolutional neural network, Briefings in Bioinformatics, 2021, 22(5):1–10.
Siguo Wang, Ying He, Zhanheng Chen, and Qinhu Zhang. FCNGRU: Locating Transcription Factor Binding Sites by Combing Fully Convolutional Neural Network With Gated Recurrent Unit, IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, 2022, 26(4):1883-1889.
