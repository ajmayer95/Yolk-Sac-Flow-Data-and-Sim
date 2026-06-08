"""Mosaic viewer subpackage — mixin classes for MosaicViewer."""

from ._types import TileInfo, TileStats, parse_tile_positions, load_graph, transform_graph_to_mosaic, _RESULT_FIELDS

import warnings as _warnings
_warnings.filterwarnings('ignore', '.*FigureCanvasAgg is non-interactive.*')
_warnings.filterwarnings('ignore', '.*Matplotlib is currently using agg.*')


def show_figure(fig=None):
    """Show a matplotlib figure in a Qt window.

    Works reliably inside napari regardless of matplotlib backend state.
    Creates a QDialog with a FigureCanvasQTAgg widget directly,
    bypassing matplotlib's backend/manager system entirely.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qtpy.QtWidgets import QDialog, QVBoxLayout
        from qtpy.QtCore import Qt

        canvas = FigureCanvasQTAgg(fig)
        dialog = QDialog()
        dialog.setWindowTitle(fig._suptitle.get_text()
                              if fig._suptitle else 'Figure')
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMinMaxButtonsHint)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        dialog.resize(int(fig.get_figwidth() * fig.dpi),
                      int(fig.get_figheight() * fig.dpi))
        canvas.draw()
        dialog.show()

        # Keep a reference so the dialog isn't garbage collected
        if not hasattr(show_figure, '_dialogs'):
            show_figure._dialogs = []
        # Prune closed dialogs
        show_figure._dialogs = [d for d in show_figure._dialogs
                                if d.isVisible()]
        show_figure._dialogs.append(dialog)

    except ImportError:
        # No Qt available — fall back to whatever matplotlib can do
        try:
            fig.show()
        except Exception:
            plt.show(block=False)


__all__ = [
    "TileInfo",
    "TileStats",
    "parse_tile_positions",
    "load_graph",
    "transform_graph_to_mosaic",
    "_RESULT_FIELDS",
]
