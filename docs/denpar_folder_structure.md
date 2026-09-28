# DenPAR folder structure

The local DenPAR copy uses one consistent directory for each official split:

```text
DenPAR Radiographs Dataset/Dataset/
|-- Training/
|   |-- Images/                    650 radiographs
|   |-- Masks (Tooth-wise)/
|   |-- Masks (Radiograph-wise)/
|   |-- Key Points Annotations/
|   `-- Bone Level Annotations/
|-- Validation/
|   |-- Images/                    150 radiographs
|   `-- ...annotations...
`-- Testing/
    |-- Images/                    200 radiographs
    `-- ...annotations...
```

`Training/Images` and `Testing/Images` are hard-linked views of the image files
from their original supplied locations. Hard links do not create another image
payload, and edits to either linked path affect the same underlying file. Treat
all DenPAR source images as read-only.

The original locations remain available for compatibility:

- Training: `dataset/raw/`
- Testing: `DenPAR Radiographs Dataset/Dataset/Images/`

For expert demonstrations, load an image from `Validation/Images` and choose
**Independent evaluation**. Do not use `dataset/test/`; it belongs to the
separate expert-validated disease-label preparation pipeline and is currently
empty.
