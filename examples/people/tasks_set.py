import numpy as np

from musket_core import tasks
from musket_core import datasources

def save_image_task(data, writer: tasks.ImageWriter):
    writer.write(data.id, data.x)

def write_image_task(data, writer: tasks.ImageWriter):
    writer.write(data.id, data.p * 255)
