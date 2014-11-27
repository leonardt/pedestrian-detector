# Pedestrian Detector
## CS194 Class Project

## Dependencies
1. cmake
2. Opencv

## Download classifier dataset

```shell
curl http://www.lookingatpeople.com/data/Daimler/pami06-munder-gavrila/DC-ped-dataset_base.tar.gz -o samples/ped-dataset.tar.gz
mkdir samples/ped-dataset
tar -xf samples/ped-dataset.tar.gz -C samples/ped-dataset
```

## Build
```shell
$ cmake .
$ make
```

## Visualize optical flow
To see the flow visualized as colors where color is a direction and intensity is
the speed
```shell
$ ./main samples/lp-right.mpeg
```

To see the flow visualized as vectors
```shell
$ ./main samples/lp-right.mpeg 1
```
