Faceprint synthetic portrait set
================================

This dataset contains 3 fictional, AI-generated portrait rows, cropped from a
single generated contact sheet. Each person folder has 8 labeled condition
crops: original, bright_light, low_light, side_pose, tilted, blurred, noisy,
and occluded.

IMPORTANT LIMITATIONS
- These are synthetic AI-generated illustrations, not photographs of real people.
- Cropping a contact sheet does not guarantee that every variation is a
  perfectly identity-consistent image of the same fictional person.
- Use this set to test file loading, face detection behavior, and demonstrate
  image-condition experiments.
- Do not use it to claim real-world recognition accuracy, fairness, security,
  or generalization. Report results explicitly as synthetic exploratory tests.
- The generated side-pose and occlusion examples may be harder for the face
  detector to process; a failed detection is itself a useful observation.

Folder layout:
  Alex_Rivera/
  Sam_Taylor/
  Taylor_Brooks/

Condition filenames:
  original.png, bright_light.png, low_light.png, side_pose.png,
  tilted.png, blurred.png, noisy.png, occluded.png
