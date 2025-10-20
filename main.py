# src/main.py
import argparse, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT


def K(N):
    k = np.arange(N).reshape(-1, 1)
    mu = np.arange(N).reshape(1, -1)
    return k @ mu


def W(N):
    return np.exp(1j * 2 * np.pi / N * K(N))


def idft(xmu):
    N = xmu.size
    return (W(N) @ xmu) / N


def fmt(A, prec=3, width=9):
    rows = [" ".join(f"{v:{width}.{prec}f}" for v in r) for r in A]
    return "[\n" + "\n".join(rows) + "\n]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument(
        "--pdf_name", default="Mateusz_Zajaczek_Variant1_IDFT_in_Matrix_Notation.pdf"
    )
    args = ap.parse_args()

    variants = {
        1: np.array([6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex),
        # w razie potrzeby dodaj kolejne
    }
    xmu = variants[args.variant]
    N = xmu.size
    Kmat, Wmat, xk = K(N), W(N), idft(xmu)
    nonzero = int(np.count_nonzero(np.abs(xmu) > 0))
    max_im = float(np.max(np.abs(np.imag(xk))))

    args.out.mkdir(parents=True, exist_ok=True)
    re_png = args.out / "variant01_Re_xk.png"
    im_png = args.out / "variant01_Im_xk.png"

    plt.figure()
    plt.stem(range(N), np.real(xk))
    plt.title(f"Re{{x[k]}} (N={N})")
    plt.xlabel("k")
    plt.ylabel("Re{x[k]}")
    plt.savefig(re_png, dpi=150, bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.stem(range(N), np.imag(xk))
    plt.title(f"Im{{x[k]}} (N={N})")
    plt.xlabel("k")
    plt.ylabel("Im{x[k]}")
    plt.savefig(im_png, dpi=150, bbox_inches="tight")
    plt.close()

    styles = getSampleStyleSheet()
    mono = ParagraphStyle(
        name="Mono",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=8.5,
        leading=10.5,
        alignment=TA_LEFT,
    )
    doc = SimpleDocTemplate(
        str(args.out / args.pdf_name),
        pagesize=A4,
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    story = []
    story += [
        Paragraph(
            "Mateusz Zajaczek – Variant 1: IDFT in Matrix Notation", styles["Title"]
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Nonzero coefficients: {nonzero}", styles["BodyText"]),
        Paragraph(f"Max |Im{{x[k]}}|: {max_im:.3e}", styles["BodyText"]),
        Spacer(1, 0.3 * cm),
        Paragraph("Matrix K = [k·μ]", styles["Heading3"]),
        Preformatted(fmt(Kmat, prec=0, width=4), mono),
        Spacer(1, 0.2 * cm),
        Paragraph("Re(W):", styles["BodyText"]),
        Preformatted(fmt(np.real(Wmat), prec=3, width=8), mono),
        Spacer(1, 0.2 * cm),
        Paragraph("Im(W):", styles["BodyText"]),
        Preformatted(fmt(np.imag(Wmat), prec=3, width=8), mono),
        Spacer(1, 0.3 * cm),
        Image(str(re_png), width=15 * cm, height=7.5 * cm),
        Spacer(1, 0.2 * cm),
        Image(str(im_png), width=15 * cm, height=7.5 * cm),
    ]
    doc.build(story)


if __name__ == "__main__":
    main()
# To run: python src/main.py --variant 1
