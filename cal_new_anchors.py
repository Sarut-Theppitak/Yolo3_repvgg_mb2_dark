old_anchors =   [
            [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj (1/8) >> most number of cells
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
                ] 
        

# new_value = 1.25 * (new_size / 448)



new_size = 384


new_anchors = [[(x * (new_size / 448), y * (new_size / 448) ) for (x,y) in ac_stride] for ac_stride in old_anchors]


for ele in new_anchors:
    print(ele)