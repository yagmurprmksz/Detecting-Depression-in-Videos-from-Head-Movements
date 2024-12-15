import itertools

class PoseKinemeHistogramTester():
    def __init__(self):
        self.opts = {
            "histogram_segment_size": [20, 40, -1],
            "selected_levels": [
                [1, 4, 7],
                [1, 4, 7, 10],
                [1, 4, 7, 10, 13],
                [4, 7, 10, 13],
            ],
            
            "hist_value": ["count", "magnitude"],
            "histogram_type": ['letter_levels', 'letter_combined', 'letter_levels_and_combined', 'kineme_levels_and_combined'],
            "kineme_type": ['single_letters', 'singleton', 'one_nod'],
        }

    def generate_combinations(self):
        # Get all combinations of the provided options
        keys, values = zip(*self.opts.items())
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def test_combinations(self):
        combinations = self.generate_combinations()
        results = []
        for combo in combinations:
            result = self.run_test(combo)
            results.append((combo, result))
        return results

    def run_test(self, config):
        # Placeholder for testing logic
        return f"Simulated result for {config}"

if __name__ == "__main__":
    tester = PoseKinemeHistogramTester()
    results = tester.test_combinations()

    # Sonuçları indeksli olarak bir dosyaya yazma
    with open("hma_kinesics/results.txt", "w", encoding="utf-8") as f:
        for i, (config, result) in enumerate(results, start=1):
            f.write(f"Index: {i}\n")
            f.write("Config:\n")
            for key, val in config.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"Result: {result}\n")
            f.write("-" * 40 + "\n\n")
