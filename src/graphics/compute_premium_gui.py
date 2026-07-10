import customtkinter as ctk

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"


class ResultCard(ctk.CTkFrame):
    """A custom frame to display a single result metric nicely."""

    def __init__(self, master, title, **kwargs):
        super().__init__(master, corner_radius=15, **kwargs)

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("gray50", "gray70")
        )
        self.title_label.pack(pady=(15, 5))

        self.value_label = ctk.CTkLabel(
            self,
            text="--",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.value_label.pack(pady=(0, 15))

    def set_value(self, value, color=None):
        if color:
            self.value_label.configure(text=value, text_color=color)
        else:
            self.value_label.configure(text=value)


class OptionsExpectancyApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Options Credit Spread / Iron Condor Calculator")
        self.geometry("850x750")
        self.minsize(750, 650)

        # --- Header ---
        self.header_label = ctk.CTkLabel(
            self,
            text="Options Expectancy Calculator",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.header_label.pack(pady=(20, 5))

        self.subheader_label = ctk.CTkLabel(
            self,
            text="Determine your minimum required credit for long-term profitability.",
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray60")
        )
        self.subheader_label.pack(pady=(0, 20))

        # --- Input Frame ---
        self.input_frame = ctk.CTkFrame(self, corner_radius=10)
        self.input_frame.pack(fill="x", padx=30, pady=10)

        # Win Rate
        self.win_rate_label = ctk.CTkLabel(self.input_frame, text="Estimated Win Rate (%):", anchor="w")
        self.win_rate_label.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        self.win_rate_entry = ctk.CTkEntry(self.input_frame, placeholder_text="e.g., 75", width=150)
        self.win_rate_entry.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="w")
        self.win_rate_entry.bind("<Return>", lambda e: self.calculate())

        # Spread Width
        self.spread_width_label = ctk.CTkLabel(self.input_frame, text="Spread Width / Max Loss ($):", anchor="w")
        self.spread_width_label.grid(row=0, column=2, padx=20, pady=20, sticky="w")
        self.spread_width_entry = ctk.CTkEntry(self.input_frame, placeholder_text="e.g., 500", width=150)
        self.spread_width_entry.grid(row=0, column=3, padx=(0, 20), pady=20, sticky="w")
        self.spread_width_entry.bind("<Return>", lambda e: self.calculate())

        # Calculate Button
        self.calc_button = ctk.CTkButton(
            self.input_frame,
            text="Calculate",
            command=self.calculate,
            width=120,
            height=35,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.calc_button.grid(row=0, column=4, padx=20, pady=20)

        # Error Label
        self.error_label = ctk.CTkLabel(self, text="", text_color="red", font=ctk.CTkFont(size=12))
        self.error_label.pack(pady=5)

        # --- Results Cards ---
        self.cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cards_frame.pack(fill="x", padx=30, pady=10)
        self.cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="card_width")

        self.card_win_rate = ResultCard(self.cards_frame, "Win Rate")
        self.card_win_rate.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.card_spread = ResultCard(self.cards_frame, "Spread Width")
        self.card_spread.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.card_breakeven = ResultCard(self.cards_frame, "Min Break-Even Credit")
        self.card_breakeven.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self.card_loss = ResultCard(self.cards_frame, "Worst-Case Loss")
        self.card_loss.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        # --- Scenario Analyzer ---
        self.scenario_outer_frame = ctk.CTkFrame(self, corner_radius=10)
        self.scenario_outer_frame.pack(fill="both", expand=True, padx=30, pady=20)

        self.scenario_title = ctk.CTkLabel(
            self.scenario_outer_frame,
            text="Scenario Analyzer (Expected Value per trade)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.scenario_title.pack(pady=(15, 10))

        self.scenario_frame = ctk.CTkScrollableFrame(self.scenario_outer_frame, corner_radius=5)
        self.scenario_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Initialize scenario headers
        self._init_scenario_headers()

        # Set initial focus
        self.win_rate_entry.focus()

    def _init_scenario_headers(self):
        headers = ["Target Profit", "Required Credit", "Actual Risk"]
        for col, header in enumerate(headers):
            lbl = ctk.CTkLabel(
                self.scenario_frame,
                text=header,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=("gray50", "gray70")
            )
            lbl.grid(row=0, column=col, padx=30, pady=10, sticky="w")

    def calculate(self):
        self.error_label.configure(text="")
        try:
            win_rate_input = float(self.win_rate_entry.get())
            if not (0 < win_rate_input < 100):
                self.error_label.configure(text="Win rate must be between 1% and 99%.")
                return

            win_rate = win_rate_input / 100.0
            spread_width = float(self.spread_width_entry.get())

            if spread_width <= 0:
                self.error_label.configure(text="Spread width must be greater than 0.")
                return

            # Core Math
            loss_rate = 1.0 - win_rate
            breakeven_credit = loss_rate * spread_width
            breakeven_actual_loss = spread_width - breakeven_credit

            # Update Cards
            self.card_win_rate.set_value(f"{win_rate_input}%")
            self.card_spread.set_value(f"${spread_width:.2f}")
            self.card_breakeven.set_value(f"${breakeven_credit:.2f}", color="blue")
            self.card_loss.set_value(f"${breakeven_actual_loss:.2f}", color="red")

            # Update Scenarios
            self.update_scenarios(breakeven_credit, spread_width)

        except ValueError:
            self.error_label.configure(text="Please enter valid numerical values.")

    def update_scenarios(self, breakeven_credit, spread_width):
        # Clear existing scenario widgets
        for widget in self.scenario_frame.winfo_children():
            widget.destroy()

        self._init_scenario_headers()

        scenarios = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100]
        for row, target_profit in enumerate(scenarios, start=1):
            req_credit = target_profit + breakeven_credit
            req_actual_loss = spread_width - req_credit

            if target_profit == 0:
                label = "Break-Even"
            else:
                label = f"+${target_profit}"

            ctk.CTkLabel(self.scenario_frame, text=label).grid(row=row, column=0, padx=30, pady=8, sticky="w")
            ctk.CTkLabel(self.scenario_frame, text=f"${req_credit:.2f}").grid(row=row, column=1, padx=30, pady=8, sticky="w")

            # Check if required credit exceeds spread width (impossible in real options)
            if req_credit > spread_width:
                risk_text = "Invalid (Credit > Width)"
                color = "red"
            else:
                risk_text = f"${req_actual_loss:.2f}"
                color = "green" if req_actual_loss > 0 else "gray"

            risk_lbl = ctk.CTkLabel(self.scenario_frame, text=risk_text, text_color=color)
            risk_lbl.grid(row=row, column=2, padx=30, pady=8, sticky="w")


if __name__ == "__main__":
    app = OptionsExpectancyApp()
    app.mainloop()