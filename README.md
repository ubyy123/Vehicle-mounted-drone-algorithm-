# Vehicle-mounted drone algorithm

![image](https://github.com/ubyy123/Vehicle-mounted-drone-algorithm-/assets/106729728/1e625136-0ede-4b3e-98b3-a3c209bae8a5)

![image](https://github.com/ubyy123/Vehicle-mounted-drone-algorithm-/assets/106729728/9cae6f9c-76d6-4539-bc0b-7887379865f8)

![image](https://github.com/ubyy123/Vehicle-mounted-drone-algorithm-/assets/106729728/830fbade-3669-4106-a952-379ab086cba8)


## Description

## Dependencies

* Python >= 3.6
* TensorFlow >= 2.0
* PyTorch = 1.5
* tqdm
* scipy
* numpy
* plotly (only for plotting)
* matplotlib (only for plotting)


## Usage

First move to `TensorFlow2` dir. 

```
cd TensorFlow2
```

Then, generate the pickle file contaning hyperparameter values by running the following command.

```
python config.py
```

you would see the pickle file in `Pkl` dir. now you can start training the model.

```
python train.py -p Pkl/***.pkl
```

Plot prediction of the pretrained model
(in this example, batch size is 128, number of customer nodes is 50)

```
python plot.py -p Weights/***.pt(or ***.h5) -b 128 -n 50
```

You can change `plot.py` into `plot_2opt.py`.  
  
2opt is a local search method, which improves a crossed route by swapping arcs.  

If you want to verify your model, you can use opensource dataset in `OpenData` dir.
  
Opensource data is obtained from Augerat et al.(1995)
  
please refer to [Capacitated VRP Instances by NEO Research Group](https://neo.lcc.uma.es/vrp/vrp-instances/capacitated-vrp-instances/)
```
python plot.py -p Weights/***.pt -t ../OpenData/A-n***.txt -b 128
```

One example would be `cd PyTorch && python plot.py -p Weights/VRP50_train_epoch19.pt -t ../OpenData/A-n45-k7.txt -d sampling -b 128` 

## Reference
* https://github.com/Rintarooo/VRP_DRL_MHA
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/d-eremeev/ADM-VRP
* https://qiita.com/ohtaman/items/0c383da89516d03c3ac0
