# Speed code
# import time
#
# # Initialize variables for speed estimation
# prev_frame_time = None
# prev_frame_positions = {}
#
# # ...
#
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#
#         # ...
#
#         # Loop over the tracks
#         for track in tracks:
#             # ...
#
#             # Calculate bounding box center
#             center = ((x1 + x2) // 2, (y1 + y2) // 2)
#
#             # Store current frame's position for speed estimation
#             prev_frame_positions[track.track_id] = center
#
#         # ...
#
#         # Calculate speed
#         if prev_frame_time is not None:
#             current_time = time.time()
#             time_diff = current_time - prev_frame_time
#
#             for track_id, prev_center in prev_frame_positions.items():
#                 if track_id in results[frame_nmr]:
#                     current_center = results[frame_nmr][track_id]['vehicle']['bbox'][:2]
#                     distance = cv2.norm(np.array(prev_center), np.array(current_center))
#                     speed = distance / time_diff
#                     results[frame_nmr][track_id]['speed estimation'] = {'speed': speed}
#
#         # Update previous frame time
#         prev_frame_time = time.time()
#         prev_frame_positions = {}
