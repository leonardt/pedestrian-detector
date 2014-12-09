import math

class Bucket:

  ACTIVE_THRESHOLD = 400
  DEACTIVE_THRESHOLD = 400
  TOLERANCE = 20

  """ presentable params:
  ACTIVE_THRESHOLD = 480 * 5
  DEACTIVE_THRESHOLD = 480 * 5
  TOLERANCE = 20
  NOISE TOLERANCE 3
  """

  """ IN angle needs to be less than OUT angle 
      in_angle:   0 - 360 degree representing in
      out_angle:  0 - 360 degrees representing out
      color:      (r, g, b) tuple representing color of bounding box"""
  def __init__(self, in_angle, out_angle, color):
    self.in_angle = in_angle
    self.out_angle = out_angle
    self.items = []
    self.bounds = []
    self.value = 0
    self.wrap = out_angle < in_angle
    self.color = color

    self.center = (self.in_angle + self.out_angle) * 0.5
    if self.wrap:
      self.center = (self.center + 180) % 360

    self.generated = False


  """ Contributes to a bin weighted by it's angle 
      relative to the center of the bin. """
  def weighted_add(self, degree, mag, old=True):

    curr_value = self.value

    in_region = (self.in_angle <= degree and degree <= self.out_angle)
    center = (self.in_angle + self.out_angle) * 0.5
    if self.wrap:
        in_region = (self.in_angle < degree and degree <= 360) or (0 <= degree and degree < self.out_angle)
        center = (center + 180) % 360

    if in_region:
      theta = math.radians(center - degree)
      if not old:
        self.value += math.cos(theta) * mag
      else:
        self.value += mag

    return (self.value - curr_value != 0)


  def commit(self):
    self.items += [self.value]
    self.value = 0

  def generate_bounds(self):
    start = 0
    active = False
    tolerance = Bucket.TOLERANCE
    for index in range(len(self.items)):
      item = self.items[index]
      if not active:
        if item > Bucket.ACTIVE_THRESHOLD:
          active = True
          start = index
      else: 
          tolerance -= 1
          if item < Bucket.DEACTIVE_THRESHOLD and tolerance <= 0:
            active = False
            self.bounds += [(start, index)]
            tolerance = Bucket.TOLERANCE
    self.generated = True


  def __getitem__(self, index):
    if not self.generated:
      self.generate_bounds()
    return self.bounds[index]



