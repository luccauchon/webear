import tkinter as tk
import numpy as np

class InteractiveCandle:
    def __init__(self, root, open, high, low, close):
        self.root = root
        self.root.title("Interactive Candlestick")

        self.canvas_width = 500
        self.canvas_height = 700

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white"
        )
        self.canvas.pack(fill="both", expand=True)

        # Initial OHLC values
        self.open = open
        self.high = high
        self.low = low
        self.close = close

        self.price_min = np.min([open, high, low, close])
        self.price_max = np.max([open, high, low, close])

        self.selected = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.draw()

    # ----------------------------
    # Price ↔ Canvas conversion
    # ----------------------------

    def price_to_y(self, price):
        margin = 50

        return (
            margin
            + (self.price_max - price)
            / (self.price_max - self.price_min)
            * (self.canvas_height - 2 * margin)
        )

    def y_to_price(self, y):
        margin = 50

        price = (
            self.price_max
            - (y - margin)
            / (self.canvas_height - 2 * margin)
            * (self.price_max - self.price_min)
        )

        return round(price, 2)

    # ----------------------------
    # Drawing
    # ----------------------------

    def draw(self):
        self.canvas.delete("all")

        x = self.canvas_width // 2
        candle_width = 80

        y_open = self.price_to_y(self.open)
        y_close = self.price_to_y(self.close)
        y_high = self.price_to_y(self.high)
        y_low = self.price_to_y(self.low)

        color = "green" if self.close >= self.open else "red"

        # Wick
        self.canvas.create_line(
            x, y_high,
            x, y_low,
            width=3,
            fill=color
        )

        # Body
        self.canvas.create_rectangle(
            x - candle_width // 2,
            y_open,
            x + candle_width // 2,
            y_close,
            fill=color,
            outline="black"
        )

        # Handles
        r = 8

        self.handles = {
            "high": (x, y_high),
            "open": (x + 60, y_open),
            "close": (x - 60, y_close),
            "low": (x, y_low),
        }

        for name, (hx, hy) in self.handles.items():

            self.canvas.create_oval(
                hx - r,
                hy - r,
                hx + r,
                hy + r,
                fill="dodgerblue",
                tags=name
            )

            self.canvas.create_text(
                hx + 25,
                hy,
                text=name.upper(),
                anchor="w",
                font=("Arial", 10, "bold")
            )

        # OHLC Display
        self.canvas.create_text(
            20,
            20,
            anchor="nw",
            font=("Consolas", 14, "bold"),
            text=(
                f"OPEN : {self.open:.2f}\n"
                f"HIGH : {self.high:.2f}\n"
                f"LOW  : {self.low:.2f}\n"
                f"CLOSE: {self.close:.2f}"
            )
        )

    # ----------------------------
    # Mouse interaction
    # ----------------------------

    def on_click(self, event):

        threshold = 15

        for name, (hx, hy) in self.handles.items():

            dx = event.x - hx
            dy = event.y - hy

            if dx * dx + dy * dy < threshold * threshold:
                self.selected = name
                return

    def on_drag(self, event):

        if not self.selected:
            return

        price = self.y_to_price(event.y)

        if self.selected == "open":
            self.open = price

        elif self.selected == "close":
            self.close = price

        elif self.selected == "high":
            self.high = price

        elif self.selected == "low":
            self.low = price

        # Maintain OHLC consistency

        self.high = max(
            self.high,
            self.open,
            self.close
        )

        self.low = min(
            self.low,
            self.open,
            self.close
        )

        self.draw()

    def on_release(self, event):
        self.selected = None


if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveCandle(root=root, open=7500, high=7600, low=7300, close=7483)
    root.mainloop()