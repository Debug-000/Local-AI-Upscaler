from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from upscaler import (
    ImageUpscaler,
    UpscaleResult,
    UpscalerError,
    enhancement_options,
    get_image_dimensions,
    supported_formats_text,
)


class RoundedPanel(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        bg_color: str,
        page_bg: str,
        radius: int = 24,
        padding: int = 22,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            bg=page_bg,
            highlightthickness=0,
            bd=0,
            relief="flat",
            **kwargs,
        )
        self.bg_color = bg_color
        self.page_bg = page_bg
        self.radius = radius
        self.padding = padding
        self.inner = tk.Frame(self, bg=bg_color)
        self.window_id = self.create_window(
            self.padding,
            self.padding,
            anchor="nw",
            window=self.inner,
        )
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, event=None) -> None:
        req_height = self.inner.winfo_reqheight() + self.padding * 2
        self.configure(height=req_height)
        self._redraw()

    def _on_canvas_configure(self, event=None) -> None:
        self.itemconfigure(self.window_id, width=max(self.winfo_width() - self.padding * 2, 1))
        self._redraw()

    def _redraw(self) -> None:
        self.delete("panel_bg")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        radius = min(self.radius, width // 2, height // 2)
        self._rounded_rect(0, 0, width, height, radius, fill=self.bg_color, outline="")
        self.tag_lower("panel_bg")

    def _rounded_rect(self, x1: int, y1: int, x2: int, y2: int, r: int, **kwargs) -> None:
        points = [
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1,
        ]
        self.create_polygon(points, smooth=True, splinesteps=24, tags="panel_bg", **kwargs)


class SoftButton(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command,
        bg_color: str,
        hover_color: str,
        text_color: str,
        disabled_bg: str,
        disabled_text: str,
        radius: int = 16,
        height: int = 46,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            height=height,
            highlightthickness=0,
            bd=0,
            relief="flat",
            bg=parent.cget("bg"),
            cursor="hand2",
            **kwargs,
        )
        self._text = text
        self._command = command
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._text_color = text_color
        self._disabled_bg = disabled_bg
        self._disabled_text = disabled_text
        self._radius = radius
        self._state = "normal"
        self._hover = False

        self.bind("<Configure>", self._redraw)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def configure(self, cnf=None, **kwargs):
        if "text" in kwargs:
            self._text = kwargs.pop("text")
        if "state" in kwargs:
            self._state = kwargs.pop("state")
        if kwargs:
            super().configure(cnf, **kwargs)
        self._redraw()

    config = configure

    def _on_click(self, _event=None) -> None:
        if self._state != "disabled" and self._command is not None:
            self._command()

    def _on_enter(self, _event=None) -> None:
        self._hover = True
        self._redraw()

    def _on_leave(self, _event=None) -> None:
        self._hover = False
        self._redraw()

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        bg = self._disabled_bg if self._state == "disabled" else self._hover_color if self._hover else self._bg_color
        fg = self._disabled_text if self._state == "disabled" else self._text_color
        self._rounded_rect(0, 0, width, height, self._radius, fill=bg, outline="")
        self.create_text(
            width // 2,
            height // 2,
            text=self._text,
            fill=fg,
            font=("Helvetica", 10, "bold"),
        )

    def _rounded_rect(self, x1: int, y1: int, x2: int, y2: int, r: int, **kwargs) -> None:
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


class SoftSwitch(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        variable: tk.BooleanVar,
        colors: dict[str, str],
    ) -> None:
        super().__init__(parent, bg=parent.cget("bg"))
        self.variable = variable
        self.colors = colors
        self.label = tk.Label(
            self,
            text=text,
            bg=parent.cget("bg"),
            fg=colors["ink"],
            font=("Helvetica", 11, "bold"),
            anchor="w",
        )
        self.label.pack(side=tk.LEFT, padx=(0, 12))

        self.switch = tk.Canvas(
            self,
            width=48,
            height=28,
            bg=parent.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.switch.pack(side=tk.LEFT)

        for widget in (self, self.label, self.switch):
            widget.bind("<Button-1>", self._toggle)

        self.variable.trace_add("write", self._redraw)
        self.switch.bind("<Configure>", self._redraw)
        self._redraw()

    def _toggle(self, _event=None) -> None:
        self.variable.set(not self.variable.get())

    def _redraw(self, *_args) -> None:
        self.switch.delete("all")
        enabled = self.variable.get()
        bg = self.colors["switch_on"] if enabled else self.colors["switch_off"]
        knob = self.colors["switch_knob"]
        self._rounded_rect(self.switch, 2, 2, 46, 26, 12, fill=bg, outline="")
        knob_x = 32 if enabled else 16
        self.switch.create_oval(knob_x - 10, 4, knob_x + 10, 24, fill=knob, outline="")

    def _rounded_rect(self, canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int, r: int, **kwargs) -> None:
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


class UpscalerUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Image Upscaler")
        self.root.geometry("1220x860")
        self.root.minsize(980, 740)

        self.colors = {
            "bg": "#0a1020",
            "surface": "#10182d",
            "surface_2": "#14203a",
            "surface_3": "#19284a",
            "surface_soft": "#0d1528",
            "surface_hover": "#202f56",
            "surface_hover_soft": "#18233f",
            "ink": "#f4f7ff",
            "muted": "#98a8c4",
            "accent": "#26385e",
            "accent_2": "#32456f",
            "accent_3": "#1d2a48",
            "input": "#0b1324",
            "warning": "#ffd166",
            "switch_on": "#31456f",
            "switch_off": "#1c2742",
            "switch_knob": "#f4f7ff",
        }

        self.upscaler = ImageUpscaler()

        self.input_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(
            value=str((Path(__file__).resolve().parent / "output").resolve())
        )
        self.scale_var = tk.IntVar(value=2)
        self.enhancement_var = tk.StringVar(value="balanced")
        self.face_restore_var = tk.BooleanVar(value=True)
        self.require_ai_var = tk.BooleanVar(value=True)
        self.original_dimensions_var = tk.StringVar(value="Original\n-")
        self.upscaled_dimensions_var = tk.StringVar(value="After\n-")
        self.status_var = tk.StringVar(
            value="Choose an image, then run the upscale to compare the result."
        )

        self.preview_canvas_width = 760
        self.preview_canvas_height = 430
        self.content_min_width = 980
        self.content_max_width = 1280
        self.preview_split = 0.5
        self.preview_placeholder = "Choose an image to start the comparison preview."
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.after_output_path: Path | None = None
        self.after_output_engine = ""
        self.after_output_face_restored = False
        self.preview_canvas: tk.Canvas | None = None
        self.preview_spinner_job: str | None = None
        self.preview_spinner_step = 0
        self.preview_generation_token = 0
        self.preview_generation_in_progress = False
        self.preview_settings_signature: tuple | None = None
        self.progress_var = tk.StringVar(value="Idle 0%")
        self.progress_percent = 0
        self.log_lines: list[str] = []
        self.progress_canvas: tk.Canvas | None = None
        self.log_text: tk.Text | None = None

        self._build_layout()
        self._bind_setting_traces()

    def _build_layout(self) -> None:
        self.root.configure(bg=self.colors["bg"])

        outer = tk.Frame(self.root, bg=self.colors["bg"])
        outer.pack(fill=tk.BOTH, expand=True)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self.scroll_canvas = tk.Canvas(outer, bg=self.colors["bg"], highlightthickness=0, bd=0)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.Scrollbar(outer, orient="vertical", command=self.scroll_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        self.page = tk.Frame(self.scroll_canvas, bg=self.colors["bg"])
        self.page.grid_columnconfigure(0, weight=1)
        self.page.grid_columnconfigure(1, weight=0)
        self.page.grid_columnconfigure(2, weight=1)
        self.page_window = self.scroll_canvas.create_window((0, 0), window=self.page, anchor="nw")

        self.page.bind("<Configure>", self._on_page_configure)
        self.scroll_canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel()

        shell = tk.Frame(self.page, bg=self.colors["bg"], padx=28, pady=28)
        shell.grid(row=0, column=1, sticky="nsew")
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_columnconfigure(1, weight=1)

        header = tk.Frame(shell, bg=self.colors["bg"])
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        tk.Label(
            header,
            text="AI Image Upscaler",
            bg=self.colors["bg"],
            fg=self.colors["ink"],
            font=("Helvetica", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Compare the source image against the latest upscale and keep the workflow simple.",
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            font=("Helvetica", 11),
        ).pack(anchor="w", pady=(6, 0))

        self._build_compare_panel(shell).grid(row=1, column=0, columnspan=2, sticky="ew")
        self._build_control_panel(shell).grid(row=2, column=0, sticky="nsew", padx=(0, 12), pady=(20, 0))
        self._build_side_panel(shell).grid(row=2, column=1, sticky="nsew", padx=(12, 0), pady=(20, 0))

    def _build_compare_panel(self, parent: tk.Frame) -> RoundedPanel:
        panel = self._panel(parent, self.colors["surface_2"], width=0)
        panel.inner.grid_columnconfigure(0, weight=1)

        self._section_header(
            panel.inner,
            "Before / After",
            "Drag the handle or use the slider to compare the source image with the latest generated result.",
        ).grid(row=0, column=0, sticky="ew")

        self.preview_canvas = tk.Canvas(
            panel.inner,
            width=self.preview_canvas_width,
            height=self.preview_canvas_height,
            bg=self.colors["input"],
            bd=0,
            highlightthickness=0,
        )
        self.preview_canvas.grid(row=1, column=0, sticky="ew", pady=(18, 14))
        self.preview_canvas.bind("<Button-1>", self._drag_split)
        self.preview_canvas.bind("<B1-Motion>", self._drag_split)

        self.split_scale = tk.Scale(
            panel.inner,
            from_=0,
            to=100,
            orient="horizontal",
            showvalue=False,
            bg=self.colors["surface_2"],
            fg=self.colors["ink"],
            troughcolor=self.colors["surface_3"],
            activebackground=self.colors["accent"],
            highlightthickness=0,
            bd=0,
            command=self._on_split_scale,
        )
        self.split_scale.set(50)
        self.split_scale.grid(row=2, column=0, sticky="ew")

        metrics = tk.Frame(panel.inner, bg=self.colors["surface_2"])
        metrics.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_columnconfigure(1, weight=1)
        self._metric_card(metrics, self.original_dimensions_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._metric_card(metrics, self.upscaled_dimensions_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self._render_compare_preview()
        return panel

    def _build_control_panel(self, parent: tk.Frame) -> RoundedPanel:
        panel = self._panel(parent, self.colors["surface"])
        panel.inner.grid_columnconfigure(0, weight=1)

        self._section_header(
            panel.inner,
            "Controls",
            "Pick the source image, choose the upscale settings, and save the result.",
        ).grid(row=0, column=0, sticky="ew")

        self._path_field(panel.inner, 1, "Input image", self.input_path_var, "Browse", self._choose_input_image)
        self._path_field(panel.inner, 2, "Output folder", self.output_dir_var, "Choose", self._choose_output_folder)

        options = tk.Frame(panel.inner, bg=self.colors["surface"])
        options.grid(row=3, column=0, sticky="ew", pady=(20, 0))
        options.grid_columnconfigure(0, weight=1)
        options.grid_columnconfigure(1, weight=1)

        self._choice_panel(
            options,
            "Scale",
            "Choose the target size.",
            [("2x", 2), ("4x", 4)],
            self.scale_var,
        ).grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._choice_panel(
            options,
            "Enhancer",
            "Apply local cleanup after the upscale.",
            [(name.capitalize(), name) for name in enhancement_options()],
            self.enhancement_var,
        ).grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ai = self._panel(panel.inner, self.colors["surface_soft"], padding=16, radius=18)
        ai.grid(row=4, column=0, sticky="ew", pady=(20, 0))
        tk.Label(ai.inner, text="AI options", bg=self.colors["surface_soft"], fg=self.colors["ink"], font=("Helvetica", 14, "bold")).pack(anchor="w")
        tk.Label(
            ai.inner,
            text="Leave AI required on if you want true super-resolution rather than a basic resize fallback.",
            bg=self.colors["surface_soft"],
            fg=self.colors["muted"],
            font=("Helvetica", 10),
            justify="left",
            wraplength=500,
        ).pack(anchor="w", pady=(6, 14))
        self._toggle(ai.inner, "Require AI backend", self.require_ai_var).pack(anchor="w")
        self._toggle(ai.inner, "Restore faces with GFPGAN", self.face_restore_var).pack(anchor="w", pady=(12, 0))

        actions = tk.Frame(panel.inner, bg=self.colors["surface"])
        actions.grid(row=5, column=0, sticky="ew", pady=(20, 0))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        self.refresh_button = self._button(actions, "Generate Preview", self._update_preview, filled=False)
        self.refresh_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.save_button = self._button(actions, "Upscale and Save", self._run_upscale, filled=True)
        self.save_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        return panel

    def _build_side_panel(self, parent: tk.Frame) -> RoundedPanel:
        panel = self._panel(parent, self.colors["surface_2"])
        panel.inner.grid_columnconfigure(0, weight=1)

        credits = self._panel(panel.inner, self.colors["surface_soft"], padding=18, radius=18)
        credits.grid(row=0, column=0, sticky="ew")
        tk.Label(
            credits.inner,
            text="Credits & Support",
            bg=self.colors["surface_soft"],
            fg=self.colors["ink"],
            font=("Helvetica", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            credits.inner,
            text="Built for local use with Real-ESRGAN, GFPGAN, Pillow, OpenCV, and Tkinter.",
            bg=self.colors["surface_soft"],
            fg=self.colors["muted"],
            font=("Helvetica", 10),
            justify="left",
            wraplength=420,
        ).pack(anchor="w", pady=(8, 0))
        tk.Label(
            credits.inner,
            text="Includes and depends on open-source work from Python, Pillow, OpenCV, PyTorch, TorchVision, BasicSR, Real-ESRGAN, and GFPGAN.",
            bg=self.colors["surface_soft"],
            fg=self.colors["muted"],
            font=("Helvetica", 10),
            justify="left",
            wraplength=420,
        ).pack(anchor="w", pady=(10, 0))
        tk.Label(
            credits.inner,
            text="If you later find this useful on GitHub, consider leaving a star and supporting the project.",
            bg=self.colors["surface_soft"],
            fg=self.colors["warning"],
            font=("Helvetica", 10, "bold"),
            justify="left",
            wraplength=420,
        ).pack(anchor="w", pady=(12, 0))

        activity = self._panel(panel.inner, self.colors["surface_soft"], padding=18, radius=18)
        activity.grid(row=1, column=0, sticky="ew", pady=(18, 0))
        tk.Label(
            activity.inner,
            text="Preview Activity",
            bg=self.colors["surface_soft"],
            fg=self.colors["ink"],
            font=("Helvetica", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            activity.inner,
            textvariable=self.progress_var,
            bg=self.colors["surface_soft"],
            fg=self.colors["muted"],
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(8, 8))
        self.progress_canvas = tk.Canvas(
            activity.inner,
            height=10,
            bg=self.colors["surface_soft"],
            highlightthickness=0,
            bd=0,
        )
        self.progress_canvas.pack(fill=tk.X)
        self.progress_canvas.bind("<Configure>", self._redraw_progress_bar)
        self.log_text = tk.Text(
            activity.inner,
            height=8,
            wrap="word",
            state="disabled",
            bg=self.colors["input"],
            fg=self.colors["muted"],
            insertbackground=self.colors["ink"],
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=14,
            highlightthickness=0,
        )
        self.log_text.pack(fill=tk.X, pady=(14, 0))
        self._append_log("Ready.")
        self._set_progress(0, "Idle")
        tk.Label(
            panel.inner,
            text=f"Formats: {supported_formats_text()}",
            bg=self.colors["surface_2"],
            fg=self.colors["muted"],
            font=("Helvetica", 10),
        ).grid(row=2, column=0, sticky="w", pady=(18, 0))
        return panel

    def _section_header(self, parent: tk.Widget, title: str, body: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=parent.cget("bg"))
        tk.Label(frame, text=title, bg=parent.cget("bg"), fg=self.colors["ink"], font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(
            frame,
            text=body,
            bg=parent.cget("bg"),
            fg=self.colors["muted"],
            font=("Helvetica", 10),
            justify="left",
            wraplength=640,
        ).pack(anchor="w", pady=(6, 0))
        return frame

    def _panel(self, parent: tk.Widget, color: str, **kwargs) -> RoundedPanel:
        return RoundedPanel(parent, bg_color=color, page_bg=self.colors["bg"], **kwargs)

    def _path_field(self, parent: tk.Frame, row: int, label: str, variable: tk.StringVar, action: str, command) -> None:
        block = tk.Frame(parent, bg=parent.cget("bg"))
        block.grid(row=row, column=0, sticky="ew", pady=(18, 0))
        block.grid_columnconfigure(0, weight=1)
        tk.Label(block, text=label, bg=parent.cget("bg"), fg=self.colors["muted"], font=("Helvetica", 10, "bold")).grid(
            row=0, column=0, sticky="w", columnspan=2, pady=(0, 8)
        )
        entry = tk.Entry(
            block,
            textvariable=variable,
            bg=self.colors["input"],
            fg=self.colors["ink"],
            insertbackground=self.colors["ink"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["surface_3"],
            highlightcolor=self.colors["accent"],
            font=("Helvetica", 10),
        )
        entry.grid(row=1, column=0, sticky="ew")
        self._button(block, action, command, filled=False).grid(row=1, column=1, padx=(10, 0))

    def _choice_panel(self, parent: tk.Frame, title: str, body: str, options: list[tuple[str, object]], variable) -> RoundedPanel:
        panel = self._panel(parent, self.colors["surface_soft"], padding=16, radius=18)
        tk.Label(panel.inner, text=title, bg=self.colors["surface_soft"], fg=self.colors["ink"], font=("Helvetica", 14, "bold")).pack(anchor="w")
        tk.Label(panel.inner, text=body, bg=self.colors["surface_soft"], fg=self.colors["muted"], font=("Helvetica", 10)).pack(anchor="w", pady=(6, 12))
        row = tk.Frame(panel.inner, bg=self.colors["surface_soft"])
        row.pack(anchor="w")
        for label, value in options:
            tk.Radiobutton(
                row,
                text=label,
                variable=variable,
                value=value,
                indicatoron=False,
                bg=self.colors["accent"],
                fg=self.colors["ink"],
                activebackground=self.colors["surface_hover"],
                activeforeground=self.colors["ink"],
                selectcolor=self.colors["surface_hover_soft"],
                relief="flat",
                bd=0,
                padx=16,
                pady=12,
                font=("Helvetica", 10, "bold"),
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(0, 10))
        return panel

    def _toggle(self, parent: tk.Widget, text: str, variable: tk.BooleanVar) -> SoftSwitch:
        return SoftSwitch(parent, text, variable, self.colors)

    def _metric_card(self, parent: tk.Frame, variable: tk.StringVar) -> RoundedPanel:
        panel = self._panel(parent, self.colors["surface_soft"], padding=16, radius=18)
        tk.Label(
            panel.inner,
            textvariable=variable,
            bg=self.colors["surface_soft"],
            fg=self.colors["ink"],
            font=("Helvetica", 14, "bold"),
            justify="left",
            anchor="w",
        ).pack(anchor="w")
        return panel

    def _button(self, parent: tk.Widget, text: str, command, filled: bool) -> SoftButton:
        return SoftButton(
            parent,
            text=text,
            command=command,
            bg_color=self.colors["accent_2"] if filled else self.colors["surface_3"],
            hover_color=self.colors["surface_hover"] if filled else self.colors["surface_hover_soft"],
            text_color=self.colors["ink"],
            disabled_bg=self.colors["surface_soft"],
            disabled_text=self.colors["muted"],
        )

    def _bind_mousewheel(self) -> None:
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.scroll_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.scroll_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _bind_setting_traces(self) -> None:
        for variable in (
            self.scale_var,
            self.enhancement_var,
            self.face_restore_var,
            self.require_ai_var,
        ):
            variable.trace_add("write", self._on_settings_changed)

    def _on_mousewheel(self, event) -> None:
        if event.num == 4:
            self.scroll_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.scroll_canvas.yview_scroll(1, "units")
        else:
            self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_page_configure(self, _event=None) -> None:
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.scroll_canvas.itemconfigure(self.page_window, width=event.width)
        target_width = max(
            self.content_min_width,
            min(self.content_max_width, event.width - 96),
        )
        self.page.grid_columnconfigure(1, minsize=target_width)

    def _on_split_scale(self, value: str) -> None:
        self.preview_split = max(0.0, min(1.0, float(value) / 100.0))
        self._render_compare_preview()

    def _drag_split(self, event) -> None:
        if self.preview_canvas is None:
            return
        width = max(self.preview_canvas.winfo_width(), 1)
        self.preview_split = max(0.0, min(1.0, event.x / width))
        self.split_scale.set(int(self.preview_split * 100))
        self._render_compare_preview()

    def _choose_input_image(self) -> None:
        path = self._native_pick_file()
        if not path:
            filetypes = [
                ("Image files", "*.png *.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
            ]
            path = filedialog.askopenfilename(title="Choose input image", filetypes=filetypes)
        if path:
            self.input_path_var.set(path)
            self.after_output_path = None
            self.after_output_engine = ""
            self.after_output_face_restored = False
            self._update_preview()

    def _choose_output_folder(self) -> None:
        path = self._native_pick_folder()
        if not path:
            path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_dir_var.set(path)

    def _native_pick_file(self) -> str:
        system = platform.system()
        start_dir = self._existing_parent(self.input_path_var.get()) or self._existing_parent(
            self.output_dir_var.get()
        )
        if system == "Windows":
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
                "$dialog.Filter = 'Image files (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg'; "
                "$dialog.Title = 'Choose input image'; "
                + (f"$dialog.InitialDirectory = '{start_dir}'; " if start_dir else "")
                + "if ($dialog.ShowDialog() -eq 'OK') { [Console]::Write($dialog.FileName) }"
            )
            return self._run_native_dialog(["powershell", "-NoProfile", "-Command", script])
        if system == "Darwin":
            script = 'POSIX path of (choose file with prompt "Choose input image")'
            return self._run_native_dialog(["osascript", "-e", script])
        if system == "Linux" and shutil.which("zenity") is not None:
            command = [
                "zenity",
                "--file-selection",
                "--title=Choose input image",
                "--file-filter=Image files | *.png *.jpg *.jpeg",
            ]
            if start_dir:
                command.append(f"--filename={start_dir}/")
            return self._run_native_dialog(command)
        return ""

    def _native_pick_folder(self) -> str:
        system = platform.system()
        start_dir = self._existing_parent(self.output_dir_var.get()) or self._existing_parent(
            self.input_path_var.get()
        )
        if system == "Windows":
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$dialog.Description = 'Choose output folder'; "
                + (f"$dialog.SelectedPath = '{start_dir}'; " if start_dir else "")
                + "if ($dialog.ShowDialog() -eq 'OK') { [Console]::Write($dialog.SelectedPath) }"
            )
            return self._run_native_dialog(["powershell", "-NoProfile", "-Command", script])
        if system == "Darwin":
            script = 'POSIX path of (choose folder with prompt "Choose output folder")'
            return self._run_native_dialog(["osascript", "-e", script])
        if system == "Linux" and shutil.which("zenity") is not None:
            command = [
                "zenity",
                "--file-selection",
                "--directory",
                "--title=Choose output folder",
            ]
            if start_dir:
                command.append(f"--filename={start_dir}/")
            return self._run_native_dialog(command)
        return ""

    def _run_native_dialog(self, command: list[str]) -> str:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError:
            return ""

        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _existing_parent(self, value: str) -> str:
        if not value:
            return ""
        path = Path(value).expanduser()
        target = path if path.is_dir() else path.parent
        return str(target) if target.exists() else ""

    def _update_preview(self) -> None:
        input_path = self.input_path_var.get().strip()
        if not input_path:
            self.original_dimensions_var.set("Original\n-")
            self.upscaled_dimensions_var.set("After\n-")
            self._clear_preview()
            self.status_var.set("Select an image to preview dimensions.")
            return

        try:
            width, height = get_image_dimensions(Path(input_path))
        except UpscalerError as exc:
            self.original_dimensions_var.set("Original\n-")
            self.upscaled_dimensions_var.set("After\n-")
            self._clear_preview()
            self.status_var.set(str(exc))
            return

        scale = self.scale_var.get()
        self.original_dimensions_var.set(f"Original\n{width} x {height}")
        if self.after_output_path and self.after_output_path.exists():
            after_width, after_height = get_image_dimensions(self.after_output_path)
            self.upscaled_dimensions_var.set(f"After\n{after_width} x {after_height}")
        else:
            self.upscaled_dimensions_var.set(f"After\n{width * scale} x {height * scale}")
        self._render_compare_preview()
        self._start_preview_generation()

    def _on_settings_changed(self, *_args) -> None:
        input_path = self.input_path_var.get().strip()
        if not input_path:
            return
        self.after_output_path = None
        try:
            width, height = get_image_dimensions(Path(input_path))
            self.original_dimensions_var.set(f"Original\n{width} x {height}")
            self.upscaled_dimensions_var.set(
                f"After\n{width * self.scale_var.get()} x {height * self.scale_var.get()}"
            )
        except UpscalerError:
            return
        self._render_compare_preview()
        self._start_preview_generation()

    def _start_preview_generation(self) -> None:
        input_text = self.input_path_var.get().strip()
        output_text = self.output_dir_var.get().strip()
        if not input_text or not output_text:
            return

        signature = (
            input_text,
            output_text,
            self.scale_var.get(),
            self.enhancement_var.get(),
            self.face_restore_var.get(),
            self.require_ai_var.get(),
        )
        self.preview_settings_signature = signature
        self.preview_generation_token += 1
        token = self.preview_generation_token
        self.preview_generation_in_progress = True
        self.status_var.set("Generating preview...")
        self._append_log("Starting preview generation.")
        self._set_progress(10, "Queued")
        self._set_loading_state(True)
        self._render_compare_preview()

        def worker() -> None:
            scale = signature[2]
            enhancement = signature[3]
            face_restore = signature[4]
            require_ai = signature[5]
            try:
                self.root.after(0, lambda: self._append_log("Preparing output path and settings."))
                self.root.after(0, lambda: self._set_progress(25, "Preparing"))
                preview_path = self._preview_output_path(Path(input_text), Path(output_text))
                self.root.after(0, lambda: self._append_log(f"Running {scale}x preview generation."))
                self.root.after(0, lambda: self._set_progress(55, "Processing"))
                result = self.upscaler.upscale_to_path(
                    input_path=Path(input_text),
                    output_path=preview_path,
                    scale=scale,
                    enhancement=enhancement,
                    face_restore=face_restore,
                    require_ai=require_ai,
                )
                self.root.after(0, lambda: self._append_log("Preview generated, updating compare view."))
                self.root.after(0, lambda: self._set_progress(85, "Finishing"))
            except Exception as exc:
                self.root.after(
                    0,
                    lambda current_token=token, current_error=exc: self._finish_preview_generation(
                        current_token,
                        error=current_error,
                    ),
                )
                return

            self.root.after(
                0,
                lambda current_token=token, current_result=result: self._finish_preview_generation(
                    current_token,
                    result=current_result,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_preview_generation(self, token: int, result=None, error: Exception | None = None) -> None:
        if token != self.preview_generation_token:
            return

        self.preview_generation_in_progress = False
        self._set_loading_state(False)

        if error is not None:
            if isinstance(error, UpscalerError):
                self.status_var.set(str(error))
            else:
                self.status_var.set(f"Preview generation failed: {error}")
            self._append_log(f"Preview failed: {error}")
            self._set_progress(0, "Failed")
            self.after_output_path = None
            self.after_output_engine = ""
            self.after_output_face_restored = False
            self._render_compare_preview()
            return

        if result is None:
            self.status_var.set("Preview generation failed: no result was returned.")
            self._append_log("Preview failed: no result returned.")
            self._set_progress(0, "Failed")
            self.after_output_path = None
            self.after_output_engine = ""
            self.after_output_face_restored = False
            self._render_compare_preview()
            return

        self.after_output_path = result.output_path
        self.after_output_engine = result.engine
        self.after_output_face_restored = result.face_restored
        self.upscaled_dimensions_var.set(f"After\n{result.output_size[0]} x {result.output_size[1]}")
        self.status_var.set("Preview generated. Save when ready.")
        self._append_log(f"Preview ready with {result.engine}.")
        self._set_progress(100, "Ready")
        self._render_compare_preview()

    def _set_loading_state(self, loading: bool) -> None:
        if loading:
            self.refresh_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            self._tick_spinner()
        else:
            self.refresh_button.configure(state="normal", text="Generate Preview")
            self.save_button.configure(state="normal")
            if self.preview_spinner_job is not None:
                self.root.after_cancel(self.preview_spinner_job)
                self.preview_spinner_job = None

    def _tick_spinner(self) -> None:
        frames = ("Generating Preview .", "Generating Preview ..", "Generating Preview ...")
        self.refresh_button.configure(text=frames[self.preview_spinner_step % len(frames)])
        self.preview_spinner_step += 1
        self.preview_spinner_job = self.root.after(250, self._tick_spinner)

    def _preview_output_path(self, input_path: Path, output_dir: Path) -> Path:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f".preview_{input_path.stem}_{self.scale_var.get()}x{input_path.suffix.lower()}"

    def _set_progress(self, percent: int, label: str) -> None:
        self.progress_percent = max(0, min(100, percent))
        self.progress_var.set(f"{label} {self.progress_percent}%")
        self._redraw_progress_bar()

    def _redraw_progress_bar(self, _event=None) -> None:
        if self.progress_canvas is None:
            return
        self.progress_canvas.delete("all")
        width = max(self.progress_canvas.winfo_width(), 1)
        self.progress_canvas.create_rectangle(0, 0, width, 10, fill=self.colors["surface_3"], outline="")
        fill_width = int(width * (self.progress_percent / 100))
        if fill_width > 0:
            self.progress_canvas.create_rectangle(0, 0, fill_width, 10, fill=self.colors["accent_2"], outline="")

    def _append_log(self, message: str) -> None:
        self.log_lines.append(message)
        self.log_lines = self.log_lines[-8:]
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", "\n".join(self.log_lines))
        self.log_text.configure(state="disabled")

    def _prepare_cover_preview(self, image_path: Path) -> Image.Image:
        width, height = self._current_preview_size()
        with Image.open(image_path) as image:
            source = image.convert("RGB")
            scale = max(width / source.width, height / source.height)
            resized = source.resize(
                (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
                Image.Resampling.LANCZOS,
            )
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    def _current_preview_size(self) -> tuple[int, int]:
        if self.preview_canvas is None:
            return self.preview_canvas_width, self.preview_canvas_height

        width = self.preview_canvas.winfo_width()
        height = self.preview_canvas.winfo_height()
        if width <= 1 or height <= 1:
            return self.preview_canvas_width, self.preview_canvas_height
        return width, height

    def _render_compare_preview(self) -> None:
        if self.preview_canvas is None:
            return

        before_path = self.input_path_var.get().strip()
        if not before_path:
            self._clear_preview()
            return

        try:
            before = self._prepare_cover_preview(Path(before_path))
            has_after = bool(self.after_output_path and self.after_output_path.exists())
            if has_after:
                after = self._prepare_cover_preview(self.after_output_path)
            else:
                width, height = self._current_preview_size()
                after = Image.new("RGB", (width, height), self.colors["input"])
        except (OSError, FileNotFoundError):
            self._clear_preview()
            return

        width, height = self._current_preview_size()
        split_x = int(width * self.preview_split)

        composite = before.copy()
        if split_x < width:
            composite.paste(after.crop((split_x, 0, width, height)), (split_x, 0))

        self.preview_photo = ImageTk.PhotoImage(composite)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_photo)
        if not has_after:
            self.preview_canvas.create_rectangle(
                split_x,
                0,
                width,
                height,
                fill="#08101d",
                outline="",
                stipple="gray50",
            )
        self.preview_canvas.create_line(split_x, 0, split_x, height, fill="white", width=2)
        self.preview_canvas.create_oval(
            split_x - 15,
            height // 2 - 15,
            split_x + 15,
            height // 2 + 15,
            fill="white",
            outline="",
        )
        self.preview_canvas.create_text(
            18,
            18,
            anchor="nw",
            text=self.original_dimensions_var.get().replace("\n", " "),
            fill=self.colors["ink"],
            font=("Helvetica", 12, "bold"),
        )
        self.preview_canvas.create_text(
            width - 18,
            18,
            anchor="ne",
            text=self.upscaled_dimensions_var.get().replace("\n", " "),
            fill=self.colors["ink"],
            font=("Helvetica", 12, "bold"),
        )
        self.preview_canvas.create_text(
            18,
            height - 18,
            anchor="sw",
            text="Before",
            fill=self.colors["ink"],
            font=("Helvetica", 11, "bold"),
        )
        self.preview_canvas.create_text(
            width - 18,
            height - 18,
            anchor="se",
            text="After" if has_after else "Generating..." if self.preview_generation_in_progress else "Run preview to generate the after view",
            fill=self.colors["ink"],
            font=("Helvetica", 11, "bold"),
        )
        if not has_after:
            self.preview_canvas.create_text(
                width - 36,
                height // 2,
                anchor="e",
                text="Generating preview..." if self.preview_generation_in_progress else "No result yet",
                fill=self.colors["ink"],
                font=("Helvetica", 20, "bold"),
            )

    def _clear_preview(self) -> None:
        if self.preview_canvas is None:
            return
        width, height = self._current_preview_size()
        self.preview_photo = None
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            width // 2,
            height // 2,
            text=self.preview_placeholder,
            fill=self.colors["muted"],
            font=("Helvetica", 18, "bold"),
            width=340,
        )

    def _run_upscale(self) -> None:
        input_text = self.input_path_var.get().strip()
        output_text = self.output_dir_var.get().strip()
        if not input_text:
            messagebox.showerror("Missing input", "Choose an input image first.")
            return
        if not output_text:
            messagebox.showerror("Missing output folder", "Choose an output folder first.")
            return

        if self.preview_generation_in_progress:
            messagebox.showinfo("Preview running", "Wait for the preview generation to finish first.")
            return

        try:
            if self.after_output_path and self.after_output_path.exists():
                final_output_path = self.upscaler.promote_preview_output(
                    self.after_output_path,
                    Path(output_text),
                    Path(input_text),
                    self.scale_var.get(),
                )
                result = UpscaleResult(
                    input_path=Path(input_text).expanduser().resolve(),
                    output_path=final_output_path,
                    scale=self.scale_var.get(),
                    enhancement=self.enhancement_var.get(),
                    face_restored=self.after_output_face_restored,
                    original_size=get_image_dimensions(Path(input_text)),
                    output_size=get_image_dimensions(final_output_path),
                    engine=self.after_output_engine or "Preview cache",
                )
            else:
                result = self.upscaler.upscale_image(
                    input_path=Path(input_text),
                    output_dir=Path(output_text),
                    scale=self.scale_var.get(),
                    enhancement=self.enhancement_var.get(),
                    face_restore=self.face_restore_var.get(),
                    require_ai=self.require_ai_var.get(),
                )
        except UpscalerError as exc:
            messagebox.showerror("Upscale failed", str(exc))
            self.status_var.set(str(exc))
            return

        self.after_output_path = result.output_path
        self.after_output_engine = result.engine
        self.after_output_face_restored = result.face_restored
        self.original_dimensions_var.set(f"Original\n{result.original_size[0]} x {result.original_size[1]}")
        self.upscaled_dimensions_var.set(f"After\n{result.output_size[0]} x {result.output_size[1]}")
        self._render_compare_preview()
        self._append_log(f"Saved final image to {result.output_path.name}.")
        self._set_progress(100, "Saved")

        enhancement_text = result.enhancement.capitalize()
        face_text = "Face restore on" if result.face_restored else "Face restore off"
        status_prefix = "Saved with AI upscale" if "Pillow fallback" not in result.engine else "Saved with fallback resize"
        self.status_var.set(
            f"{status_prefix}. {enhancement_text} enhancer. {face_text}. Saved to {result.output_path}"
        )
        messagebox.showinfo(
            "Upscale complete",
            (
                f"Saved upscaled image to:\n{result.output_path}\n\n"
                f"Backend: {result.engine}\n"
                f"Enhancement: {enhancement_text}\n"
                f"Face restoration: {'Enabled' if result.face_restored else 'Disabled'}\n"
                f"Output size: {result.output_size[0]} x {result.output_size[1]}"
            ),
        )
