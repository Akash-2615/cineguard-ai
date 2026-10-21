# Near-infrared path

## What a retroreflector does

A phone lens is a small optical system. An illuminator mounted next to the camera sends light out. Most surfaces scatter it. A lens aimed back at the camera returns a tight beam into the lens that sent it. On the sensor that return is a compact, saturated spot with a bright centre and a dark ring around it. A lit phone screen is the opposite: it is a wide emitter, and a display panel emits very little near-infrared, so the same spot should not appear from the screen itself.

That is why a dark auditorium is a poor place for ordinary colour CCTV and a plausible place for a co-located infrared illuminator.

## What this repository actually does

`NIR_MODE` defaults to `simulated`. `NIRSimulator` mixes the colour frame with a heavy red weight, applies gamma, and can add a radial falloff so the frame looks a little like a ring of LEDs around the lens. The wavelength control is a label, 850 or 940. It does not change the pixels. There is no optical filter and no illuminator in software that matches a real one.

`NIR_MODE=off` scores pose only and caps the score so it cannot reach an alert unless pose-only alerts are turned on. `NIR_MODE=real` skips the colour mix and expects the incoming frame to already be mono or NIR.

The glint detector runs inside a padded hand box. It looks for a compact, round, high-contrast peak. Large, flat, rectangular regions are rejected, including a bright rectangle that stands in for a lit screen. A spot must show up in several recent frames near the hand before it is confirmed.

The synthetic injector is off unless a NIR Lab toggle or a manifest row enables it. It paints a Gaussian at the hand centre and every resulting detection is stored with `synthetic: true`.

## Assumptions that a real camera would break

- The red channel is only a stand-in for an IR-pass image.
- Vignette is a radial gradient, not the beam of a real LED ring.
- Exposure, gain, and flicker are whatever the colour camera already did.
- A projector flash or a shiny watch can still fool the shape rules. The tests cover a rectangle and a noise field, not a theatre.
- Seat mapping and pose are ordinary 2D vision. A phone facing the screen and hidden from this camera is invisible.

## Validation plan once hardware exists

1. Mount an 850 nm or 940 nm illuminator beside the lens. Add an IR-pass filter. Prefer a global shutter and lock the exposure.
2. Set `NIR_MODE=real`. Turn synthetic glint off.
3. Record the four scripted cases in [../demo_clips/README.md](../demo_clips/README.md) under that lighting, plus a lit-screen negative and a glasses/watch negative.
4. Retune percentile, area, circularity, contrast, and persistence on those clips. Save the config snapshot with the report.
5. Publish the clip list, the hardware, and the report together. Until that pack exists, do not describe the current scores as an accuracy claim.

## Hardware checklist

- 850 nm or 940 nm illuminator next to the lens
- IR-pass filter
- Global shutter
- Exposure lock
- The same camera position used for the seat-corner calibration
