import os
import math
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
import numpy as np
import cv2
import json
from collage import make_collage

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

#----------------------------------------------------------------------------------
# Utility functions

def calculate_clusters(pixels, cluster_count=5):
	"""
	Use the pixel data from the image to calculate clusters of pixel colors
	"""

	# Set the random seed so we don't get randomized clusters
	np.random.seed(100)
	
	# Convert it to a NumPy array
	# (what most datascience libraries need instead of a list)
	np_pixels = np.float32(pixels)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	
	# Apply KMeans
	compactness,labels,centers = cv2.kmeans(np_pixels,cluster_count,None,criteria,1,cv2.KMEANS_PP_CENTERS)
	# Convert back to tuples
	clusters = [tuple([round(c) for c in center]) for center in centers]
	print(str(clusters))
	return clusters
	


def hex_to_tuple(hex):
	""" Convert a hex color #FF0000 to a tuple (255,0,0)"""
	assert len(hex) == 7 and hex[0] == "#", f"Is this actually a hex value: '{hex}'"

	x = (int(hex[1:3],16),int(hex[3:5],16),int(hex[5:7],16))
	return x

#----------------------------------------------------------------------------------
# Loading the dataset

def load_color_dataset(path):
	"""
	Load the path to a JSON dataset
	and convert each entry into a Color instance
	Parameters:
		path
	Returns:
		list of Color
	"""

	# Task 0:
	# Load the XKCD color dataset.  
	# Review how to open JSON data.
	# Find the part of the dataset with the *list of colors dicts*
	# (use some_data.keys() to list all the keys in a dictionary)
	
	#  Use a list comprehension to return it as 
	#  a list of Color instances instead
	file = open(path)
	color_dataset = json.load(file)
	print(color_dataset)
	list_of_colors = color_dataset['colors']
	if list_of_colors != []:	
		return 	[Color(color) for color in list_of_colors]
	return []


#----------------------------------------------------------------------------------
# Query functions for finding colors


def get_colors_by_word(all_colors, word):
	"""
	Given a color name, return all entries that contain that color in all_colors
	ie: "purple" will find "royal purple" "light purple" etc.  
	Return them *sorted by length of name 
	
	Parameters: 
		all_colors (list of Color):  a list of Color objects
		name(str): a name of a color
	Returns:
		List of Color: all Color that
		None: if that color is not found
	"""

	# Task 1: get color by word
	# In this assignment we will sort several things using a function
	# For this one, here's a helper function already written
	# (read on sorting with a "key function")
	# https://www.programiz.com/python-programming/methods/list/sort
	list = [color for color in all_colors if word in color.name]
	def name_length(color):
		return color.name
	list.sort(key=name_length)
	if list != []:
		return list
	return None

def get_closest_named_colors(all_colors, query_rgb, count=10):

	"""
	Parameters: 
		all_colors (list of Color)
	Returns:
		list of Color: the top N colors with the smallest distance to the color rgb
	"""

	# Task 2
	# Don't just get a single close color, get the top ten (or 20)!
	# The usual store-the-winner approach doesn't work here, because 
	# we need to have multiple winners.

	# In this function, we will try an alternative approach:
	# * first sort *all* the colors by proximity to the query color
	#	* In Python we can use the "sorted" function to sort a list
	#	* but we can also pass it a function to use when determining how to sort a list!
	#		* sorted(some_list, key=some_comparison_function)
	#		BE SURE TO USE SORTED (which makes a copy) 
	#		NOT SORT which changes the original list
	#		https://www.programiz.com/python-programming/methods/built-in/sorted
	# * Now that the list is sorted, return a slice of the top N colors

	# Note that this is much more convenient to program, but slower 
	# (we are sorting thousands of colors, only to throw out 99%!)
	def distance(color):
		return color.get_distance_to(query_rgb)
	sorted_list = sorted(all_colors, key=distance)
	return sorted_list[0:count]

#---------------------------------------------------------------------------------
# Drawing utility functions
		

def draw_scatterplot(axes, colors, marker="o"):
	""" 
	Draws a 3D scatterplot of these RGB values, on these axes, 
	using the color, and using the given marker
	# https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
	# https://www.youtube.com/watch?v=PnwpoCDA5IM

	Parameters: 
		axes: the Matplotlib axes to add data to
		colors: list of tuples
		marker: "^", "o" or other Matplotlib marker styles (sets the shape)
	Returns 
		none
	"""


	# Task 4
	# Use "axes.scatter" to draw each color in the list
	# To set the color parameter of "scatter", the rgb values 
	# need to be [0-1] instead if [0-255]
	# Use a list comprehension to make an "rgb_1" copy of each color, 
	# with its values divided by 255
	rgb_1_list = [(color[0]/255,color[1]/255,color[2]/255) for color in colors]
	for rgb_1_color in rgb_1_list:
		axes.scatter(rgb_1_color[0],rgb_1_color[1],rgb_1_color[2], marker=marker)
	return None


def draw_colors(image, rect_size=(100, 200), rgb_colors=[], labels=None, font=None):
	"""
	Draws rectangles or labels on an image using a list of colors (rgb tuples)
	
	Draw a column of rectangles or labels 
	like this, in each color, starting from the top left corner (0,0)
	filled with each color in the list
	If there are labels, draw text instead of rectangles (also filled with the right color)

	 ____
	|___|
	|___|
	|___|
	|___|
	|___|	

	Returns None
	"""

	# Task 5: 
	# Implement draw_colors, which will help us visualize sets of colors
	# You will need to use "rectangle" and "text" commands in the image draw library
	# https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
	
	# For each color
	# 	Draw either a rectangle or text (if there are labels)

	# 	Use a for loop with enumerate, because we need to know
	# 	 the number of each rectangle
	# 	- so we can create two tuples that hold its top-left/bottom-right points
	# 	- and add the right label (at the same position)

	# (look at 'main' for how to draw rectangles)

	# Create the drawing tool
	im_draw = ImageDraw.Draw(image)  
	top_left = (0,0)
	bottom_right = rect_size
	if labels != None:
		for i, rgb_color in enumerate(rgb_colors):
			im_draw.text([top_left[0],top_left[1], bottom_right[0],bottom_right[1]], labels[i], fill=rgb_color)
			top_left = (0,top_left[1]+rect_size[1])
			bottom_right = (rect_size[0], bottom_right[1]+rect_size[1])
	else:
		for i, rgb_color in enumerate(rgb_colors):
			im_draw.rectangle([top_left[0],top_left[1], bottom_right[0],bottom_right[1]], fill=rgb_color)
			top_left = (0,top_left[1]+rect_size[1])
			bottom_right = (rect_size[0], bottom_right[1]+rect_size[1])
	return None

def sort_cats(all_cats):
	"""
	Parameters:
		all_cats: list of CatPicture
	Returns
		List of CatPicture, sorted by the number of cats found, most-to-least
	"""

	# Task 10
	# Sort the cat list, by the number of cats found
	# Use the other two sorting tasks as a template!
	# Edge case: If a CatPicture does not have the attribute detected_cats, 
	#	consider it to have 0 cats
	def number_of_cats_found(cat):
		if hasattr(cat, "detected_cats"):
			return len(cat.detected_cats)
		else: 
			return 0
	all_cats.sort(reverse = True, key=number_of_cats_found)
	return all_cats[:]


class Color: 
	"""
	Color class
	A useful way to store colors in several formats
	"""

	def __init__(self, xkcd_entry):
		""" Initialize from the XKCD entry """

		assert "hex" in xkcd_entry, "Wrong datatype passed to Color constructor"
		self.name = xkcd_entry["color"]
		self.hex = xkcd_entry["hex"]
		self.rgb = hex_to_tuple(self.hex)

	def __str__(self):
		""" Stringify """

		return f"{self.name} {self.rgb}"

	def get_distance_to(self, rgb):
		""" Calculate the distance between this color and another RGB tuple
		as a pythagorean distance between the vectors of their RGB points
		We can use this as a measurement of how similar the colors are

		Parameters:
			rgb (tuple of ints)
		Returns 
			float (distance between RGB values)
		"""
		assert type(rgb) == tuple and len(rgb) == 3, f"expected tuple of len 3, got {rgb}" 


		# Task 3
		# Each RGB value is a point in 3D space
		# Calculate the distance between the two points

		# Compute d_red, d_green, and d_blue values that are the difference 
		# from the rgb values to *this* color's rgb values
		# 
		# Add their squares 
		# 	(the python syntax for raising something to a power is x**2) 
		# and return the square root 
		# 	"math.sqrt")
		self_tuple = self.rgb
		d_red = self_tuple[0]-rgb[0]
		d_green = self_tuple[1]-rgb[1]
		d_blue = self_tuple[2]-rgb[2]
		distance = math.sqrt(d_red**2 + d_green**2+ d_blue**2)
		return distance


class Cat_Picture:
	"""
	A class to represent cat pictures
	Each picture contains an image
	But also can contain lots of interesting data about that image
	including:
		* the aspect ratio of this image
		* the clusters of all the pixel color
		* "palette", the names of all the colors

	Attributes:
		file_path: the Path for this image's file
		full_image (PIL Image): the full-size image of this cat picture (in RGB)
		width (int): the width of the full image in pixels
		height (int): the height of the full image in pixels
		aspect_ratio (int): width/height of the image
			Aspect ratios > 1 are *wide*, < 1 are *tall*
		thumbnail (PIL Image): the smaller version of the picture 
			(for when we want fewer pixels)

	"""
	def __init__(self, file_path):

		# Task 6:
		# Load a full-size PIL Image for this image and convert it to RGB
		# https://realpython.com/image-processing-with-the-python-pillow-library/
		# Store attributes width, height, full_image, aspect_ratio, and file_path
		# 	for this instance
		
		# --- your code here ---
		self.file_path = file_path
		with Image.open(file_path) as img:
			img.load()
		self.full_image = img.convert("RGB")
		self.width = img.width
		self.height = img.height
		self.aspect_ratio = self.width / self.height
		self.thumbnail = self.create_thumbnail(200)
		return None

		
	def create_thumbnail(self, thumbnail_base=200):
		"""
		Make a *tiny* copy and shrink it to a thumbnail
		This gives us smaller pixel data to work with
		"""

		# To get the width and height of the thumbnail
		# 	*multiply* the base size by the squareroot of the aspect ratio
		# 	*divide* the base size by the squareroot of the aspect ratio
		# The size should be a tuple of integers, so use "round" to round the answers
		# https://www.w3schools.com/python/ref_func_round.asp


		resampling_technique = Image.Resampling.LANCZOS

		# Task 7: 
		# Change the thumbnail size
		thumbnail_size = (100,100)
		
		# Use "thumbnail" to make a smaller version of this image
		# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail
		# Note that "thumbnail" *replaces* an image with a smaller version
		# so first make a *copy* 
		# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.copy
		copy = self.full_image.copy()
		copy.thumbnail(thumbnail_size, resample=resampling_technique)
		return copy

	def set_palette(self, named_colors,cluster_count=7):
		"""
		Calculate the RBG clusters based on the thumbnail
		use the named colors to convert those RGB values into 
		 the closest named Color instances

		Parameters:
			named_colors: List of Color
			seed: int
			cluster_count: int
		Returns:
			None
		"""
		
		# Task 8
		# Calculate the clusters for this image by clustering the pixels 
		# 	from the thumnail image, to get a list of rgb tuples
		# Use get_closest_named_colors to turn that into a list of Color
		# (great use of a list comprehension!)

		# Get the pixel data
		pixels = self.thumbnail.getdata()

		# --- your code here ---
		clusters = calculate_clusters(pixels, cluster_count = cluster_count)
		# for cluster in clusters:
		# 	self.palette = get_closest_named_colors(named_colors, cluster, cluster_count)
		self.palette = [get_closest_named_colors(named_colors, cluster)[0] for cluster in clusters]
		return None


	def get_pixels(self):
		return list(self.thumbnail.getdata())


	def detect_cats(self):
		""" 
		Detect the boxes containing cats or humans
		There are machine learning models that perform better at this task
		But this "cascade" type model is from 2016 and is very easy to set up
		https://www.geeksforgeeks.org/face-detection-using-cascade-classifier-using-opencv-python/
		"""

		# The path to the model
		cascade_path = "ml_models/haarcascade_frontalcatface.xml"

		# Convert the PIL image to an array of numbers
		npimage = np.array(self.full_image) 
		cv2.cvtColor(npimage, cv2.COLOR_RGB2GRAY)
		
		# Perform cat detection
		detector = cv2.CascadeClassifier(cascade_path)
		rects = detector.detectMultiScale(npimage, scaleFactor=1.06, minSize=(60,60),minNeighbors=1)

		
		
		self.detected_cats = rects

	def draw_detected_cats(self):
		"""
		Returns a copy of the full image, but with all cats 
		detected shown as white rectangles
		"""

		# Task 8
		# Detect and draw cats

		# No cats yet? Try to detect some
		# Use hasattr to detect if this class has "detected_cats" as an attribute
		# https://www.programiz.com/python-programming/methods/built-in/hasattr
		# - if not, call "detect_cats" to detect cats
		
		# Next, draw the cats on a *copy* of the full image
		# then return the copy
		# - Create a copy, an ImageDraw tool, and draw a rectangle
		# - for each cat detected, with a white outline of width 10
		# 		Review the rectangle drawing documentation!
		# 		The boxes that cat-detection gives us are [x, y, width, height]
		# 		which is not the same format as the rectangles, 
		# 		so you will have to convert it (draw it out on paper!)
		# https://pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/
		
		copy = self.full_image.copy()
		
		# --- your code here ---
		if not hasattr(self, "detected_cats"):
			copy.detected_cats = self.detect_cats()

		copy.detected_cats = self.detected_cats

		im_draw = ImageDraw.Draw(copy)

		for cat in self.detected_cats:
			x = cat[0]
			y = cat[1]
			w = cat[2]
			h = cat[3]
			im_draw.rectangle([x, y, x + w, y + h], outline="white", width=10)
		return copy


	#-------------------------------------
	# Cat visualization	

	def create_scatterplot(self, ax):
		# Use the draw_scatterplot to draw this cat as a scatterplot of pixels

		# This confusing code adds the thumbnail image to the graph
		im = OffsetImage(self.thumbnail, zoom=.2)
		ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction')
		ax.add_artist(ab)

		# Get every 300th color
		colors = self.get_pixels()[::300]
		draw_scatterplot(ax, colors)
		

	def print_data(self):
		print(f"\nCat picture: {self.file_path}")

		print(f"\tsize:{self.width}x{self.height} pixels")
		print(f"\taspect_ratio:{self.aspect_ratio:.2f}")
		
		if self.thumbnail:
			print(f"\tthumbnail:{self.thumbnail.size[0]}x{self.thumbnail.size[1]} pixels")
		else:
			print("\tno thumbnail")

		print(f"\tmode:{im.mode}")

#===============================================================================
# Main
#===============================================================================

if __name__ == "__main__":

	def color_list_to_string(colors):
		# Stringify and join a list of colors
		if len(colors) == 0:
			return "<empty list>"
		return ", ".join([str(c) for c in colors])

	#===============================================================================
	# Test Task 0, 1, 2 - Finding colors
	
	print("\n" + "-"*80 + "\nFinding colors")
	
	# Test Task 0 by loading all the XKCD colors
	xkcd_colors = load_color_dataset("xkcd-colors.json")
	print(f"Loaded {len(xkcd_colors)} colors")

	first_colors = xkcd_colors[0:5]
	print("\nFirst colors: ", color_list_to_string(first_colors))

	assert first_colors[0].name == "cloudy blue"
	assert len(xkcd_colors) == 949, "We expected 949 colors"
	
	# Test Task 1
	blues = get_colors_by_word(xkcd_colors, "blue")
	violets = get_colors_by_word(xkcd_colors, "violet")
	aquas = get_colors_by_word(xkcd_colors, "aqua")
	neons = get_colors_by_word(xkcd_colors, "neon")
	slimes = get_colors_by_word(xkcd_colors, "slime")
	print("Violets found: ", color_list_to_string(violets))
	print("Aquas found: ", color_list_to_string(aquas))
	print("Slimes found: ", color_list_to_string(slimes))

	assert aquas[0].name == "aqua", f"'aqua' should be first, as it is shortest"


	# # Test Task 2 and 3

	blue = blues[0]
	blue_RGB = (0, 0, 255)
	red_RGB = (255, 0, 0)
	blue_distance = blue.get_distance_to(blue_RGB)
	print("\nTesting distance")
	print(f"\tDistance from {blue} to blue {blue_RGB}: {blue_distance:.2f}")
	red_distance = blue.get_distance_to(red_RGB)
	print(f"\tDistance from {blue} to red {red_RGB}: {red_distance:.2f}")
	#assert math.isclose(blue_distance, 74.31, rel_tol=.1), f"\t{blue} distance to {blue_RGB}: {blue_distance:.2f}, expected 74.31"


	# # What color is Northwestern purple, really?

	# # #401f68 is the color taken from northwestern.edu's background
	# # so it must be the official Northwestern purple
	nu_purple_rgb = hex_to_tuple("#401f68") 
	nu_many_purples = get_closest_named_colors(xkcd_colors, nu_purple_rgb, 12)
	print("\nNU's purple is", color_list_to_string(nu_many_purples))

	assert nu_many_purples[0].name == "royal purple", f"Royal purple should be the closest, you had {nu_many_purples[0]}"
	assert nu_many_purples[9].name == "grape", f"Grape should be the 9th, you had {nu_many_purples[9]}"
	assert len(nu_many_purples) == 12, f"Make sure you find the right number of colors"


	# # How far off is the Northwestern purple from "royal purple"?
	print(f"\nTesting distance to {nu_purple_rgb}: ")
	for color in nu_many_purples:
		distance = color.get_distance_to(nu_purple_rgb)
		print(f"\t{color} distance to {nu_purple_rgb}: {distance:.2f}")

	royal = nu_many_purples[0]
	royal_distance = royal.get_distance_to(nu_purple_rgb)
	assert math.isclose(royal_distance, 33.44, rel_tol=.1), f"\t{royal} distance to {nu_purple_rgb}: {royal_distance:.2f}, expected 33.44"

	# #===============================================================================
	# # DRAWING AND GRAPHS

	print("\n" + "-"*80 + "\nDrawing colors")

	# # -- MATPLOTLIB --
	# # Use Matplotlib to make a 3D scatter graph 
	# # Matplotlib is a great way to show interactive graphs with Python
	# # (** though you need to close it manually when you are done **)

	# # Remember that RGB colors are a point in 3D space
	# # So we can draw them in a 3D graph
	# # **Note: this can take a minute to display**


	fig = plt.figure()
	axes = fig.add_subplot(projection='3d')

	# # Task 
	# # Try turning these on and off to see different colors
	# draw_scatterplot(axes, [c.rgb for c in aquas], marker="^")
	# draw_scatterplot(axes, [c.rgb for c in blues], marker="*")
	# draw_scatterplot(axes, [c.rgb for c in violets], marker="o")

	# # All colors
	draw_scatterplot(axes, [c.rgb for c in xkcd_colors], marker="s")
	axes.set_xlabel('red')
	axes.set_ylabel('green')
	axes.set_zlabel('blue')	
	# plt.show()


	# # -- PIL DRAWING --
	# # We will be using the Python image library PIL/Pillow

	# # Read this article first!
	# # https://realpython.com/image-processing-with-the-python-pillow-library/

	# # If we want to see this color we can *draw it into an image*
	# # ** Notice ** PIL often wants data as a *tuple* 
	# #  (e.g. points and colors and dimensions)

	# # Create the image first
	image_dimensions = (300, 200)
	im = Image.new(mode="RGB", size=image_dimensions)

	# # Create a drawing tool
	im_draw = ImageDraw.Draw(im)  

	# # Use the drawing tool to draw a rectangle 
	# #   using a list of its top-left/bottom-right corners
	# #	and an RGB tuple for the color
	# # In graphics, its *very* useful to diagram on paper or whiteboard!
	rect_corners = [(0,0), (150, 90)]
	im_draw.rectangle(rect_corners, fill=nu_purple_rgb)

	# # This displays the image
	# # You have to manually close it, so comment these out when you don't need them
	# im.show()

	# #---------------------------------------------
	# # # *** More drawing practice ***
	
	# # Practice drawing rectangles and text until you understand how to draw them
	# # in different colors and positions
	some_color = (230, 5, 255)
	im_draw.rectangle([(100,100), (200, 200)], fill=some_color)

	# # Load a font so that we can add text
	font = ImageFont.truetype(font="FredokaOne-Regular.ttf", size=18)

	# # Add text with a black outline
	im_draw.text((0, 0), "Go Wildcats!", font=font, 
		fill=(155, 55, 255), stroke_fill=(0,0,0),stroke_width=2)
	
	# # Show the image
	# im.show()

	# # We can sample colors from this test image too
	test_pixel_color = im.getpixel((0, 0))
	print(f"original color: {nu_purple_rgb}")
	print(f"color of the test rectangle {test_pixel_color}")

	# # Test Task 5
	# # Implement draw_colors function to draw a column of rectangles or text
	# # We can use this to visualize data later

	colors = [(255,0,0), (0,255,0), (0,0,255)]
	labels = ["red", "green", "blue"]

	# # No labels just rectangles
	draw_colors(im, rect_size=(100, 30), rgb_colors=colors)
	# # No rectangles just labels
	draw_colors(im, rect_size=(100, 30), rgb_colors=colors, labels=labels, font=font)
	im.show()

	# # You should see three rectangles labeled "red" "green" and "blue"
	# # We can sample the image inside the rectangles to see if they are the right color
	assert im.getpixel((50,2)) == (255,0,0), "Is the top row red?"
	assert im.getpixel((50,32)) == (0,255,0), "Is the middle row green?"
	assert im.getpixel((50,62)) == (0,0,255), "Is the bottom row blue?"

	# #===============================================================================
	# # Cat Pictures

	print("\n" + "-"*80 + "\nCat Pictures")

	# # Images!
	# # Let's load an image for practice
	
	cat_path = os.path.join("images", "cat0.png")
	
	with Image.open(cat_path) as im:

	# 	# We want to work with this image in RGB mode, so we convert it
		im = im.convert('RGB')

		print(f"\nLoaded image: {cat_path}")
		print(f"\ttype: {type(im)}")
		print(f"\tsize:{im.size[0]}x{im.size[1]} pixels")
		print(f"\tmode:{im.mode}")

	# 	# PIP also has filters
	# 	# Try turning these on and off
	# 	# Blur
		# im = im.filter(ImageFilter.GaussianBlur(20)).show()
	# 	# Convert to gray and find edges
		# im = im.convert("L").filter(ImageFilter.FIND_EDGES)

	# 	# "show" is a method that opens a new window to show the image
	# 	# We won't see the image UNTIL this command is called
	# 	# 	Note that all the things we have done to this image
	# 	#	up to this point are visible, but not the things we do afterwards
	# 	#	Try playing the blur filter above and below the "show command"
		
	# 	# You also need to manually close the image window.
	# 	# That can be annoying so comment out
	# 	# 	any "show" commands you aren't using!

		# im.show()

	# 	# Try opening a few other cat images by changing the file path above
	# 	# Can you find a large or small one?
	
	# # Test the CatPicture class
	# # Test Task 6 and 7
	cat_path = os.path.join("images", "cat0.png")
	
	# # Test with a very wide cat (turn off for asserts)
	# # cat_path = os.path.join("images", "cat67.png")
	
	cat_picture = Cat_Picture(cat_path)
	cat_picture.print_data()

	# # Show this cat
	cat_picture.full_image.show()
	
	# # Check the height, width, and aspect ratio
	# # (for cat0, it will be different numbers for the others)
	assert cat_picture.width == 600 
	assert cat_picture.height == 447 

	assert math.isclose(cat_picture.aspect_ratio, 1.34, abs_tol=.02)

	thumbnail_ratio = cat_picture.thumbnail.size[0]/cat_picture.thumbnail.size[1]
	print("Thumbnail aspect:" ,thumbnail_ratio)

	# # We should still have the same aspect ratio for a thumbnail and full size
	# # Make sure the images look the same, but one is *smaller*
	cat_picture.full_image.show()
	cat_picture.thumbnail.show()
	print("Thumbnail pixel count = ", cat_picture.thumbnail.size[0]*cat_picture.thumbnail.size[1])
	print("Full-size pixel count = ", cat_picture.full_image.size[0]*cat_picture.full_image.size[1])
	assert math.isclose(cat_picture.aspect_ratio, thumbnail_ratio, abs_tol=.02)
	assert cat_picture.full_image.size[0] > cat_picture.thumbnail.size[0], "Make sure you are not replacing the original cat picture with the thumbnail, but are making a copy"
	
	# # Another cat picture, should have *different* data
	cat_path2 = os.path.join("images", "cat2.png")
	cat_picture2 = Cat_Picture(cat_path2)
	cat_picture2.print_data()
	cat_picture2.thumbnail.show()
	assert cat_picture.full_image.size[0] != cat_picture2.full_image.size[0], "Make sure you are loading the path passed in as a parameter"
	
	
	# # Now we can use this to visualize color data!
	# # Get the pixel data from the thumbnail and turn it into a list
	cat_pixel_colors = list(cat_picture.thumbnail.getdata())	
	# # Look at every 300th pixel, and draw it as a skinny stripe 
	# # This style is sometimes used to visualize movie color palettes
	# # https://happycoding.io/gallery/movie-colors/index
	# # and gives us an overall sense of the colors
	cat_pixel_colors = cat_pixel_colors[::200]
	draw_colors(im, rect_size=(300, 1), rgb_colors=cat_pixel_colors)
	im.show()
	
	
	print("\n" + "-"*80 + "\nCat Picture Pixels!\n")

	# # Colors and drawing rectangles
	# # Ok, so we have an image, which is a big list of RGB colors
	# # We can access and display these colors in different ways

	# # Lets use the cat thumbnail as our test image
	# # and get the list of all its pixels
	test_image = cat_picture.thumbnail
	pixels = list(test_image.getdata())
	
	# # Here are all the pixels in the image.  Beautiful, right?
	# print(pixels)

	# # .. and just the one in the top left
	# print(pixels[0])


	# We can get individual pixels from some point in the image
	# with (x,y) coordinates
	# Here is the upper left hand corrner of the image
	sample_point = (5,5)
	# And the bottom right (uncomment to use this instead)
	# sample_point = (200,100)

	
	one_pixel_color = test_image.getpixel(sample_point)
	# print("one pixel", one_pixel_color)


	# # We can also make a copy of the cat picture and draw over it
	# # for a visualization that shows colors AND the original photo
	cat_copy = cat_picture.full_image.copy()
	draw_colors(cat_copy, rect_size=(300, 1), rgb_colors=cat_pixel_colors)

	# # Make sure you see the rectangles on here, too!
	cat_copy.show()	


	# # #--------------------------------------------------------
	# # # How else can we visualize all the pixels?
	# # We can re-use your scatterplot!

	# # # How many cats to show?
	cat_count = 9
	# Which cat to start at (change this to see different cats)
	cat_offset = 22

	# Load several cats
	several_cats = [Cat_Picture(os.path.join("images", f"cat{i}.png")) 
		for i in range(cat_offset, cat_count + cat_offset)]
	
	cols = 3
	rows = cat_count//3
	# Make a 3x3
	figure, axis = plt.subplots(rows, cols, subplot_kw=dict(projection='3d'))

	for i in range(0, len(several_cats)):
		# Where in the plot does cat's subplot this go?
		x = i//cols
		y = i%cols
		ax = axis[x, y]
	
		several_cats[i].create_scatterplot(ax)

	plt.show()

	# #--------------------------------------------------------
	# # Task 8: calculating the palette

	# # Those plots were neat, but its hard to do statistics on them
	# # Can you see that some of the graphs have *clusters* of pixels?
	# # It would be neat to identify particular groups of colors that are similar
	# # K-means clustering does that!
	# # https://towardsdatascience.com/k-means-clustering-explain-it-to-me-like-im-10-e0badf10734a

	# # # We can calculate the "clusters" with this function
	# # # to get a smaller set of "representative colors" for this image
	# clusters = calculate_clusters(pixels)
	# print("Clusters found", clusters)
	# We can calculate the "clusters" with this function
 	# to get a smaller set of "representative colors" for this image
	pixels = cat_picture.thumbnail.getdata()
	clusters = calculate_clusters(pixels, cluster_count=10)
	print("Clusters found", clusters)
	
	# # And display them with our draw_colors again!
	# # Try changing the number of clusters.  
	# # What is a good number of clusters for this image?
	# cat_copy = cat_picture.full_image.copy()
	# draw_colors(cat_copy, rect_size=(300, 40), rgb_colors=clusters)
	# cat_copy.show()

	# # But we would rather have *named* colors, so that we can describe 
	# # this palette to someone else ("its a lot of greys and blues")
	# # Implement set_palette to get the palette of Color instances
	# # that best represent this image 
	# # (you may get a different set of clusters, 
	# # I haven't figured out how to make it deterministic)
	cat_picture.set_palette(xkcd_colors, cluster_count=7)
	print(color_list_to_string(cat_picture.palette))
	palette_rgb = [c.rgb for c in cat_picture.palette]
	palette_names = [c.name for c in cat_picture.palette]
	draw_colors(cat_copy, rect_size=(300, 40), rgb_colors=palette_rgb, labels=palette_names, font=font)
	cat_copy.show()

	# # #--------------------------------------------------------
	# # Test Task 9: Cat detection
	# # Implement draw_detected_cats
	cat_diagram = cat_picture.draw_detected_cats()
	cat_diagram.show()

	print("\nDetecting multiple cats on many photos")
	# # Notice that it can't find all cats

	# # How many cats to show? 
	# # Reduce if you have a slow computer, or increase to see more
	cat_count = 25
	# Which cat to start at (change this to see different cats)
	cat_offset = 0

	several_cats = [Cat_Picture(os.path.join("images", f"cat{i}.png")) 
		for i in range(cat_offset, cat_count + cat_offset)]

	# # Test Task 10
	[c.detect_cats() for c in several_cats]
	sorted_cats = sort_cats(several_cats)

	# Make a collage of all the detected cat photos
	collage = make_collage((900, 900), [cat.draw_detected_cats() for cat in sorted_cats])
	collage.show()

	assert len(sorted_cats[0].detected_cats) > len(sorted_cats[-1].detected_cats), "sort most-to-lease"




