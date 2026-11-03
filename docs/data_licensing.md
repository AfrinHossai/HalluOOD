# HalluOOD data licensing and attribution

**Verification date:** 2026-11-02. This note records the license statements found in the cited official project sources. It is a research-project summary, not legal advice or a determination of rights in any particular file.

## Scope and license boundaries

The HalluOOD repository contains original project code and documentation, while its experiments use third-party code, annotations, question/label files, images, and model weights. A license for one of these components does not automatically apply to the others.

| Material | Verified statement | Practical boundary |
| --- | --- | --- |
| Original HalluOOD code | The root [`LICENSE`](../LICENSE) and `pyproject.toml` declare MIT. | This is the license for HalluOOD's own software. It does not grant rights to POPE files, COCO assets, model weights, or other third-party material. |
| POPE repository source code | The official POPE [`LICENSE` at the pinned source revision](https://github.com/RUCAIBox/POPE/blob/08d957b917e5a378a2f99d35b6293c536a66298b/LICENSE) says MIT and names Yifan Li, copyright 2024. The grant applies to “software and associated documentation files”; it requires retaining its copyright and permission notice in copies or substantial portions of that software. | This supports use and redistribution of POPE source code subject to the MIT notice. It does not, by itself, clearly establish the license of generated data files. |
| POPE-generated questions and labels | The pinned [POPE README](https://github.com/RUCAIBox/POPE/blob/08d957b917e5a378a2f99d35b6293c536a66298b/README.md) describes the repository as providing code and data, gives the generated question-file structure, and says its COCO object annotations are used to build POPE. The inspected README and LICENSE do not state a distinct, explicit license for the released question/label files. | **Unresolved.** Public availability and the repository's software MIT license are not enough to conclude that the question/label files, or a selected/modified subset of them, may be redistributed. Their derivation from COCO annotations may also require attention to the underlying annotation terms. Obtain clarification before redistributing these files. |
| COCO annotations | The [official COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm) state that the annotations and website are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The license requires appropriate credit, a link to the license, and an indication of changes; it does not imply endorsement. | Attribute the COCO annotations and identify modifications or subsets when reusing them. The annotation license does not grant copyright in the images. |
| COCO images | The official COCO terms state that the COCO Consortium does not own the image copyrights and that image use must abide by [Flickr's terms](https://www.flickr.com/help/terms/) and the applicable image-source rights. Flickr's [Creative Commons page](https://www.flickr.com/creativecommons/) displays multiple license types; this does not establish which license applies to any particular COCO image. | Image rights are not a blanket CC BY 4.0 grant. Review the source and license applicable to each image for the intended use. Do not redistribute image bytes unless those terms permit it or the rights holder grants permission. |

The [COCO terms page](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm) also describes a separate license for COCO software. That software license is not the license for either the images or the annotations.

## Attribution practice

For research reuse, cite the POPE paper and identify the official repository and pinned release. If POPE source code is copied or redistributed, preserve its MIT copyright and permission notice. This scholarly attribution does not resolve the missing data-file license.

For COCO annotation-derived material, cite Lin et al. (2014), link to the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/), and state what was selected, transformed, or otherwise changed. The project uses a subset; any public distribution should identify the subset and its processing. Where an output includes material derived from the annotations, confirm the license obligations for that specific use rather than assuming that its POPE origin supersedes the COCO terms.

For any COCO image reused publicly, retain the image's source/creator information when available, link to the source and applicable image license, and describe modifications such as cropping or synthetic corruption. Check the specific source terms before publication or redistribution; an attribution statement alone does not create permission where the applicable terms do not grant it.

Relevant scholarly references:

- Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., & Wen, J.-R. (2023). [Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/). EMNLP 2023.
- Lin, T.-Y. et al. (2014). [Microsoft COCO: Common Objects in Context](https://doi.org/10.1007/978-3-319-10602-1_48). ECCV 2014.

## Remaining permissions and checks

Before redistributing POPE question/label files, including a derived HalluOOD subset, ask the POPE authors or repository maintainers to identify the applicable data license and confirm redistribution/adaptation rights. Also determine and document whether the released files incorporate COCO annotation material in a way that carries CC BY 4.0 obligations for the planned distribution.

Before redistributing COCO images or image-containing derivatives, check the rights and terms for each image at its source (the POPE records identify image names, but this note is not an image-by-image rights audit). If the applicable license does not cover the intended use, permission must come from the relevant rights holder. Publishing numeric measurements or code without image bytes is a separate question and should not be treated as permission to publish the underlying image files.

The project currently keeps downloaded datasets, images, model weights, caches, and generated experiment outputs outside version control. This repository policy limits accidental inclusion; it does not change third-party licenses or authorize later redistribution.

## Sources reviewed

- POPE authors' [official repository](https://github.com/RUCAIBox/POPE), [pinned README](https://github.com/RUCAIBox/POPE/blob/08d957b917e5a378a2f99d35b6293c536a66298b/README.md), and [pinned LICENSE](https://github.com/RUCAIBox/POPE/blob/08d957b917e5a378a2f99d35b6293c536a66298b/LICENSE).
- COCO Consortium's [Terms of Use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm), the [CC BY 4.0 deed](https://creativecommons.org/licenses/by/4.0/), [Flickr Terms of Use](https://www.flickr.com/help/terms/), and Flickr's [Creative Commons license directory](https://www.flickr.com/creativecommons/).
- HalluOOD's root [`LICENSE`](../LICENSE) and [`pyproject.toml`](../pyproject.toml).
