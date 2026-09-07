(Yachachiq pen plotter - 50 mm calibration square)
(Set origin at bottom-left with the pen touching the paper, then run.)
(Measure the drawn square. Each side should be 50 mm.)
G21
G90
G92 X0 Y0 Z0
G1 Z5 F300
G1 X10 Y10 F1500
G1 Z0 F300
G1 X60 Y10 F800
G1 X60 Y60 F800
G1 X10 Y60 F800
G1 X10 Y10 F800
G1 Z5 F300
G1 X0 Y0 F1500
