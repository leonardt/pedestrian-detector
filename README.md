# Pedestrian Detector
## CS194 Class Project

## Dependencies
1. cmake
2. Opencv

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
