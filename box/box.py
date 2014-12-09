import cv2
from bucket import Bucket
import os
file_path = os.path.dirname(os.path.realpath(__file__))

rgb = cv2.imread(file_path + "/output.jpg")
hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

horiz_hist = []

LEFT = Bucket(90, 270, (255, 0, 0))
RIGHT= Bucket(270, 90, (0, 0, 255))


buckets = [LEFT, RIGHT]

width = hsv.shape[1]
height = hsv.shape[0]

NOISE_THRESHOLD = 30
NOISE_GAIN = 0.1

BLACK_THRESHOLD = NOISE_THRESHOLD * 1.25
MIN_HEIGHT = 60

for x in range(width):
    for y in range(height):
        hue = 2 * hsv[y, x, 0]
        val = hsv[y, x, 2]
        for bucket in buckets:
            if val > NOISE_THRESHOLD:
                value = val
            else:
                value = val * NOISE_GAIN
            bucket.weighted_add(hue, value, False)

    for bucket in buckets:
        bucket.commit()

offset = 0
FIXED_OFFSET = 1

"""
for bucket in buckets:
    for bound in bucket:
        start, end = bound
        cv2.rectangle(rgb, (start, 5+offset), (end, 475 -offset), bucket.color, 1)
        offset += FIXED_OFFSET

"""

VERTICAL_TOLERANCE = 30

for bucket in buckets:
    for bound in bucket:
        start, end = bound

        active = False
        tolerance = VERTICAL_TOLERANCE
        for y in range(height):
            val = max(hsv[y, x, 2] for x in range(start, end))
            if active and val > BLACK_THRESHOLD:
                tolerance = VERTICAL_TOLERANCE
            if not active and val > BLACK_THRESHOLD:
                active = True
                start_y = y
            if active and val < BLACK_THRESHOLD:
                tolerance -= 1
                if tolerance <= 0:
                    active = False
                    if (y - start_y) > MIN_HEIGHT:
                        cv2.rectangle(rgb, (start, start_y), (end, y), bucket.color, 1)
                        tolerance = VERTICAL_TOLERANCE







"""
for x in range(hsv.shape[1]):
    left_bkt = 0
    right_bkt = 0

    # amt = 0
    NOISE_THRESHOLD = 3
    for y in range(hsv.shape[0]):
        hue = hsv[y, x, 0]
        val = hsv[y, x, 2]
        if hue > 0 and hue < 180:
            if val > NOISE_THRESHOLD:
                left_bkt += val
        else:
            if val > NOISE_THRESHOLD:
                right_bkt += val


        # amt += hsv[y, x, 2]
    horiz_hist.append((left_bkt, right_bkt))

left_active = False
right_active = False

left_start = 0
right_start = 0
TOLERANCE = 20

ACTIVE_TOLERANCE = 480 * 5
DEACTIVE_TOLERANCE = 480 * 5
left_tolerance = TOLERANCE
right_tolerance = TOLERANCE
columns = []
for index, buckets in enumerate(horiz_hist):
    left = buckets[0]
    right = buckets[1]


    if not left_active:
        if left > ACTIVE_TOLERANCE:
            left_active = True
            left_start = index
    else:
        left_tolerance -= 1
        if left < DEACTIVE_TOLERANCE and left_tolerance <= 0:
            left_active = False
            columns.append((left_start, index, "left"))
            left_tolerance = TOLERANCE



    if not right_active:
        if right > ACTIVE_TOLERANCE:
            right_active = True
            right_start = index
    else:
        right_tolerance -= 1
        if right < DEACTIVE_TOLERANCE and right_tolerance <= 0:
            right_active = False
            columns.append((right_start, index, "right"))
            right_tolerance = TOLERANCE

for col in columns:
    if col[2] == "left":
        cv2.rectangle(rgb, (col[0], 5+offset), (col[1], 475-offset), (0, 0, 255), 1)
        offset += FIXED_OFFSET
    elif col[2] == "right":
        cv2.rectangle(rgb, (col[0], 5+offset), (col[1], 475-offset), (255, 0, 0), 1)
        offset += FIXED_OFFSET
"""


cv2.imshow('frame1', rgb)
k = cv2.waitKey() & 0xff
