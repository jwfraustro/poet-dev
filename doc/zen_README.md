### ZEN
> Zen Eliminates Noise

This is an implementation of Drake Deming's Pixel-Level Decorrelation (PLD).

### Team Members:
* [Ryan Challener](https://github.com/rychallener/) (UCF) <rchallen@knights.ucf.edu>
* Andrew Foster (UCF)
* Em DeLarme (UCF)
* Zacchaeus Scheffer (UCF)

To clone the repo:
```shell
  git clone --recursive https://github.com/rychallener/ZEN zen
```

You have to compile the [MCcubed](https://github.com/pcubillos/MCcubed) package:
```shell
  cd zen
  cd mccubed
  make
  cd ..
```

Edit zen.cfg to your liking. At the very least, you need to specify the location of the files to be read (poetdir, cent, phot) and eventname which is the event code (e.g. wa029bs11).

you can execute with zen.py from the shell with
```shell
  zen.py <rundir> <config>
```

