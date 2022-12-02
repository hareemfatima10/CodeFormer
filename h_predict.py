import tempfile
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def set_realesrgan():
    if not torch.cuda.is_available():  # CPU
        import warnings

        warnings.warn(
            "The unoptimized RealESRGAN is slow on CPU. We do not use it. "
            "If you really want to use it, please modify the corresponding codes.",
            category=RuntimeWarning,
        )
        upsampler = None
    else:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=True,
        )
    return upsampler

def predict(input_img):
    """Load the model into memory to make running multiple predictions efficient"""
    device = "cuda:0"
    upsampler = set_realesrgan()
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt_path = "/content/drive/MyDrive/FaceSwap-Kevin/Image_restoration_weights/codeformer.pth"
    checkpoint = torch.load(ckpt_path)[
        "params_ema"
    ]  # update file permission if cannot load
    net.load_state_dict(checkpoint)
    net.eval()

    # take the default setting for the demo
    has_aligned = False
    only_center_face = False
    draw_box = False
    detection_model = "retinaface_resnet50"

    face_helper = FaceRestoreHelper(
        upscale_factor=2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )
    #default parameters from demo
    codeformer_fidelity = 0.7
    bg_upsampler = None
    face_upsample = True
    
    face_upsampler = upsampler if face_upsample else None

    #img = cv2.imread(str(input_img), cv2.IMREAD_COLOR)
    img = input_img

    if has_aligned:
        # the input faces are already cropped and aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        # get face landmarks for each face
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        print(f"\tdetect {num_det_faces} faces")
        # align and warp each face
        face_helper.align_warp_face()

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(
            cropped_face / 255.0, bgr2rgb=True, float32=True
        )
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(
                    cropped_face_t, w=codeformer_fidelity, adain=True
                )[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f"\tFailed inference for CodeFormer: {error}")
            restored_face = tensor2img(
                cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
            )

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    # paste_back
    if not has_aligned:
        # upsample the background
        if bg_upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = bg_upsampler.enhance(img, outscale=upscale_factor)[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        if face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=draw_box,
                face_upsampler=face_upsampler,
            )
        else:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=draw_box
            )
    # # save restored img
    # out_path = Path(tempfile.mkdtemp()) / 'output.png'
    imwrite(restored_img, '/content/output.png')
    return restored_img
    
def image_enhancer(input_images):
  enhanced_images = []
  for index, tup in enumerate(input_images):
    res = predict(tup[0])
    enhanced_images.append(res)
  return enhanced_images
