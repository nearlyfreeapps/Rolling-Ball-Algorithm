Rolling Ball Algorithm
----------------------

Ported to Python from ImageJ's Background Subtractor.
Only works for 8-bit greyscale images currently.
Does not perform shrinking/enlarging for larger radius sizes.

Based on the concept of the rolling ball algorithm described
in Stanley Sternberg's article,
"Biomedical Image Processing", IEEE Computer, January 1983.

Imagine that the 2D grayscale image has a third (height) dimension by the image
value at every point in the image, creating a surface. A ball of given radius
is rolled over the bottom side of this surface; the hull of the volume
reachable by the ball is the background.

http://rsbweb.nih.gov/ij/developer/source/ij/plugin/filter/BackgroundSubtracter.java.html
