from __future__ import annotations

import tkinter as tk
import warnings
from tkinter import messagebox

warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated since 0.13.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13.*",
    category=UserWarning,
)

from ui import UpscalerUI


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f"Tkinter could not start: {exc}") from exc

    try:
        UpscalerUI(root)
        root.mainloop()
    except Exception as exc:  # pragma: no cover - GUI-level safety net
        if root.winfo_exists():
            try:
                messagebox.showerror("Application error", str(exc))
            except tk.TclError:
                pass
        raise


if __name__ == "__main__":
    main()
