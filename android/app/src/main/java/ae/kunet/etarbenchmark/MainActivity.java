package ae.kunet.etarbenchmark;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.Debug;
import android.os.PowerManager;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.FileProvider;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.TimeZone;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    static {
        System.loadLibrary("etar_jni");
    }

    private static final String PREFS = "benchmark_config_v1";
    private static final String[] DISTRIBUTION_LABELS = {
        "Dense Gaussian",
        "Sparse Gaussian (10% retained)",
        "Student-t df=3 + 35% dropout",
        "Norm-heavy (log-normal row scale)"
    };
    private static final String[] DISTRIBUTIONS = {
        "dense_gaussian", "sparse10", "mixed_dropout_heavytail", "norm_heavy"
    };
    private static final String[] ORDER_LABELS = {
        "Exact f32 → Full int8 → ETAR → ETAR-LM",
        "Full int8 → ETAR → ETAR-LM → Exact f32",
        "ETAR → ETAR-LM → Exact f32 → Full int8",
        "ETAR-LM → Exact f32 → Full int8 → ETAR"
    };
    private static final String[] ORDER_VALUES = {
        "exact,full_i8,etar,etar_lm",
        "full_i8,etar,etar_lm,exact",
        "etar,etar_lm,exact,full_i8",
        "etar_lm,exact,full_i8,etar"
    };

    private final Map<String, EditText> fields = new LinkedHashMap<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private SharedPreferences preferences;
    private Spinner distribution;
    private Spinner methodOrder;
    private CheckBox quantizeQuery;
    private TextView status;
    private TextView results;
    private Button runButton;
    private Button resetButton;
    private Button shareButton;
    private File lastResultFile;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        setContentView(buildContent());
    }

    private View buildContent() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(18), dp(18), dp(32));
        scroll.addView(root);

        TextView title = text("ETAR Edge Benchmark", 24, Typeface.BOLD);
        root.addView(title);
        TextView intro = text(
            "Runs exact float32, exhaustive int8, ETAR, and ETAR-LM in the same native ARM64 library. " +
            "Settings are saved automatically. The populated defaults match the paper's representative operating point.",
            15, Typeface.NORMAL);
        intro.setPadding(0, dp(8), 0, dp(14));
        root.addView(intro);

        section(root, "Dataset and workload");
        addInteger(root, "corpus_size", "Corpus rows (N)", "50000");
        addInteger(root, "dimension", "Vector dimension (d)", "256");
        addInteger(root, "query_count", "Distinct queries", "1000");
        addInteger(root, "warmup_queries", "Warm-up queries", "100");
        addInteger(root, "repetitions", "Minimum repetitions", "1");
        addInteger(root, "seed", "Random seed", "123");

        distribution = spinner(root, "Synthetic distribution", DISTRIBUTION_LABELS,
            preferences.getInt("distribution_index", 0));

        section(root, "Search parameters");
        addInteger(root, "k", "Top K", "10");
        addInteger(root, "rerank", "Rerank candidates (R)", "100");
        addDecimal(root, "rho", "Retained query energy (ρ)", "0.90");
        addInteger(root, "head_cap", "Maximum selected coordinates", "128");
        addDecimal(root, "lambda_tail", "Tail λ", "0.75");
        addDecimal(root, "alpha_min", "Tail α minimum", "0.05");
        addDecimal(root, "alpha_max", "Tail α maximum", "0.50");

        quantizeQuery = new CheckBox(this);
        quantizeQuery.setText("Quantize query to int8 during candidate scan");
        quantizeQuery.setChecked(preferences.getBoolean("quantize_query", false));
        quantizeQuery.setPadding(0, dp(4), 0, dp(4));
        root.addView(quantizeQuery);

        section(root, "Measurement controls");
        addDecimal(root, "measurement_seconds", "Minimum trace duration per method (seconds)", "10");
        addInteger(root, "cpu_core", "CPU core (-1 lets Android schedule)", "-1");
        methodOrder = spinner(root, "Method execution order", ORDER_LABELS,
            preferences.getInt("method_order_index", 0));
        TextView hint = text(
            "Use 0 seconds for quick checks. For power-rail measurement, use 30–60 seconds and repeat with rotated method orders.",
            13, Typeface.ITALIC);
        hint.setPadding(0, dp(5), 0, dp(12));
        root.addView(hint);

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setGravity(Gravity.CENTER_VERTICAL);
        runButton = button("Run benchmark");
        runButton.setOnClickListener(v -> startBenchmark());
        actions.addView(runButton, new LinearLayout.LayoutParams(0, dp(52), 1));
        resetButton = button("Reset defaults");
        resetButton.setOnClickListener(v -> {
            preferences.edit().clear().apply();
            recreate();
        });
        LinearLayout.LayoutParams resetParams = new LinearLayout.LayoutParams(0, dp(52), 1);
        resetParams.setMarginStart(dp(8));
        actions.addView(resetButton, resetParams);
        root.addView(actions);

        shareButton = button("Share latest JSON result");
        shareButton.setEnabled(false);
        shareButton.setOnClickListener(v -> shareResult());
        LinearLayout.LayoutParams shareParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp(50));
        shareParams.topMargin = dp(8);
        root.addView(shareButton, shareParams);

        status = text("Ready", 15, Typeface.BOLD);
        status.setPadding(0, dp(16), 0, dp(8));
        root.addView(status);
        results = text("No result yet.", 13, Typeface.MONOSPACE.getStyle());
        results.setTypeface(Typeface.MONOSPACE);
        results.setTextIsSelectable(true);
        root.addView(results);
        return scroll;
    }

    private void section(LinearLayout root, String label) {
        TextView view = text(label, 18, Typeface.BOLD);
        view.setPadding(0, dp(14), 0, dp(4));
        root.addView(view);
    }

    private void addInteger(LinearLayout root, String key, String label, String defaultValue) {
        addField(root, key, label, defaultValue,
            InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED);
    }

    private void addDecimal(LinearLayout root, String key, String label, String defaultValue) {
        addField(root, key, label, defaultValue,
            InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
    }

    private void addField(LinearLayout root, String key, String label,
                          String defaultValue, int inputType) {
        TextView caption = text(label, 14, Typeface.NORMAL);
        caption.setPadding(0, dp(5), 0, 0);
        root.addView(caption);
        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setInputType(inputType);
        input.setText(preferences.getString(key, defaultValue));
        input.setSelectAllOnFocus(true);
        root.addView(input, new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));
        fields.put(key, input);
    }

    private Spinner spinner(LinearLayout root, String label, String[] values, int selection) {
        TextView caption = text(label, 14, Typeface.NORMAL);
        caption.setPadding(0, dp(7), 0, 0);
        root.addView(caption);
        Spinner spinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
            android.R.layout.simple_spinner_item, values);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setSelection(Math.max(0, Math.min(selection, values.length - 1)));
        root.addView(spinner, new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));
        return spinner;
    }

    private TextView text(String value, int sp, int style) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(sp);
        view.setTypeface(Typeface.DEFAULT, style);
        return view;
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        return button;
    }

    private void startBenchmark() {
        try {
            final Config c = readConfig();
            saveConfig();
            runButton.setEnabled(false);
            resetButton.setEnabled(false);
            shareButton.setEnabled(false);
            status.setText("Running native benchmark… Keep the device idle.");
            results.setText("The display stays on while native query batches execute.");
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            final long pssBeforeKb = Debug.getPss();
            final int thermalBefore = thermalStatus();
            final BatterySnapshot batteryBefore = batterySnapshot();
            executor.execute(() -> {
                try {
                    String nativeJson = runNativeBenchmark(
                        c.corpusSize, c.dimension, c.queryCount, c.warmupQueries,
                        c.repetitions, c.k, c.rerank, c.rho, c.headCap,
                        c.lambdaTail, c.alphaMin, c.alphaMax, c.seed,
                        c.measurementSeconds, c.quantizeQuery, c.cpuCore,
                        c.distribution, c.methodOrder);
                    long pssAfterKb = Debug.getPss();
                    int thermalAfter = thermalStatus();
                    BatterySnapshot batteryAfter = batterySnapshot();
                    String envelope = resultEnvelope(nativeJson, pssBeforeKb, pssAfterKb,
                        thermalBefore, thermalAfter, batteryBefore, batteryAfter);
                    File file = saveResult(envelope);
                    String summary = summarize(envelope, file);
                    runOnUiThread(() -> finishSuccess(file, summary));
                } catch (Throwable error) {
                    runOnUiThread(() -> finishFailure(error));
                }
            });
        } catch (Throwable error) {
            finishFailure(error);
        }
    }

    private Config readConfig() {
        Config c = new Config();
        c.corpusSize = parseLong("corpus_size");
        c.dimension = parseInt("dimension");
        c.queryCount = parseInt("query_count");
        c.warmupQueries = parseInt("warmup_queries");
        c.repetitions = parseInt("repetitions");
        c.seed = parseLong("seed");
        c.k = parseInt("k");
        c.rerank = parseInt("rerank");
        c.rho = parseFloat("rho");
        c.headCap = parseInt("head_cap");
        c.lambdaTail = parseFloat("lambda_tail");
        c.alphaMin = parseFloat("alpha_min");
        c.alphaMax = parseFloat("alpha_max");
        c.measurementSeconds = parseFloat("measurement_seconds");
        c.cpuCore = parseInt("cpu_core");
        c.quantizeQuery = quantizeQuery.isChecked();
        c.distribution = DISTRIBUTIONS[distribution.getSelectedItemPosition()];
        c.methodOrder = ORDER_VALUES[methodOrder.getSelectedItemPosition()];
        return c;
    }

    private int parseInt(String key) {
        return Integer.parseInt(value(key));
    }

    private long parseLong(String key) {
        return Long.parseLong(value(key));
    }

    private float parseFloat(String key) {
        return Float.parseFloat(value(key));
    }

    private String value(String key) {
        String value = fields.get(key).getText().toString().trim();
        if (value.isEmpty()) throw new IllegalArgumentException(key + " cannot be empty");
        return value;
    }

    private void saveConfig() {
        SharedPreferences.Editor edit = preferences.edit();
        for (Map.Entry<String, EditText> item : fields.entrySet()) {
            edit.putString(item.getKey(), item.getValue().getText().toString().trim());
        }
        edit.putInt("distribution_index", distribution.getSelectedItemPosition());
        edit.putInt("method_order_index", methodOrder.getSelectedItemPosition());
        edit.putBoolean("quantize_query", quantizeQuery.isChecked());
        edit.apply();
    }

    private int thermalStatus() {
        PowerManager power = (PowerManager) getSystemService(POWER_SERVICE);
        return power == null ? -1 : power.getCurrentThermalStatus();
    }

    private BatterySnapshot batterySnapshot() {
        BatteryManager battery = (BatteryManager) getSystemService(BATTERY_SERVICE);
        BatterySnapshot value = new BatterySnapshot();
        if (battery != null) {
            value.capacityPercent = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
            value.currentNowMicroamps = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
            value.charging = battery.isCharging();
        }
        return value;
    }

    @SuppressWarnings("unused") // Called synchronously from the JNI trace hooks.
    private long readBatteryEnergyNwh() {
        try {
            BatteryManager battery = (BatteryManager) getSystemService(BATTERY_SERVICE);
            return battery == null ? Long.MIN_VALUE
                : battery.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER);
        } catch (RuntimeException unavailable) {
            return Long.MIN_VALUE;
        }
    }

    private String resultEnvelope(String nativeJson, long pssBeforeKb, long pssAfterKb,
                                  int thermalBefore, int thermalAfter,
                                  BatterySnapshot batteryBefore,
                                  BatterySnapshot batteryAfter) throws Exception {
        JSONObject device = new JSONObject();
        device.put("manufacturer", Build.MANUFACTURER);
        device.put("model", Build.MODEL);
        device.put("device", Build.DEVICE);
        device.put("hardware", Build.HARDWARE);
        device.put("build_fingerprint", Build.FINGERPRINT);
        device.put("soc_manufacturer", Build.VERSION.SDK_INT >= 31 ? Build.SOC_MANUFACTURER : "unavailable");
        device.put("soc_model", Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : "unavailable");
        device.put("android_release", Build.VERSION.RELEASE);
        device.put("sdk", Build.VERSION.SDK_INT);
        device.put("supported_abis", new JSONArray(Build.SUPPORTED_ABIS));
        device.put("available_processors", Runtime.getRuntime().availableProcessors());
        device.put("pss_before_kb", pssBeforeKb);
        device.put("pss_after_kb", pssAfterKb);
        device.put("thermal_status_before", thermalBefore);
        device.put("thermal_status_after", thermalAfter);
        device.put("battery_capacity_before_percent", batteryBefore.capacityPercent);
        device.put("battery_capacity_after_percent", batteryAfter.capacityPercent);
        device.put("battery_current_before_microamps", batteryBefore.currentNowMicroamps);
        device.put("battery_current_after_microamps", batteryAfter.currentNowMicroamps);
        device.put("charging_before", batteryBefore.charging);
        device.put("charging_after", batteryAfter.charging);

        JSONObject envelope = new JSONObject();
        envelope.put("created_utc", utcTimestamp());
        envelope.put("app_version", "0.2.0");
        envelope.put("device", device);
        envelope.put("benchmark", new JSONObject(nativeJson));
        return envelope.toString(2);
    }

    private File saveResult(String json) throws Exception {
        File directory = new File(getExternalFilesDir(null), "results");
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IllegalStateException("could not create results directory");
        }
        File file = new File(directory, "etar-" + fileTimestamp() + ".json");
        try (FileOutputStream stream = new FileOutputStream(file)) {
            stream.write(json.getBytes(StandardCharsets.UTF_8));
        }
        return file;
    }

    private String summarize(String envelope, File file) throws Exception {
        JSONObject root = new JSONObject(envelope);
        JSONObject benchmark = root.getJSONObject("benchmark");
        JSONArray methods = benchmark.getJSONArray("methods");
        StringBuilder out = new StringBuilder();
        out.append("Completed\n\n");
        for (int i = 0; i < methods.length(); ++i) {
            JSONObject method = methods.getJSONObject(i);
            JSONObject latency = method.getJSONObject("latency_ms");
            out.append(method.getString("name"))
                .append("\n  recall=").append(String.format(Locale.US, "%.4f", method.getDouble("recall_at_k")))
                .append("  median=").append(String.format(Locale.US, "%.3f ms", latency.getDouble("median")))
                .append("  p95=").append(String.format(Locale.US, "%.3f ms", latency.getDouble("p95")))
                .append("\n  representation=").append(formatBytes(method.getLong("representation_bytes")))
                .append("  build=").append(String.format(Locale.US, "%.1f ms", method.getDouble("build_ms")));
            if (!method.isNull("energy_joules_per_query")) {
                out.append("\n  energy=")
                    .append(String.format(Locale.US, "%.6f J/query",
                        method.getDouble("energy_joules_per_query")))
                    .append(" (")
                    .append(String.format(Locale.US, "%.3f J total",
                        method.getDouble("energy_joules")))
                    .append(", ")
                    .append(String.format(Locale.US, "%.3f W avg",
                        method.getDouble("average_power_watts")))
                    .append(")");
                if (!method.isNull("idle_adjusted_energy_joules_per_query")) {
                    out.append("\n  idle-adjusted=")
                        .append(String.format(Locale.US, "%.6f J/query",
                            method.getDouble("idle_adjusted_energy_joules_per_query")))
                        .append(" (")
                        .append(String.format(Locale.US, "%.3f W",
                            method.getDouble("idle_adjusted_average_power_watts")))
                        .append(")");
                }
            } else {
                out.append("\n  energy=unavailable (")
                    .append(method.getString("energy_counter_status"))
                    .append(")");
            }
            out.append("\n\n");
        }
        JSONObject device = root.getJSONObject("device");
        out.append("Process PSS: ").append(device.getLong("pss_before_kb"))
            .append(" → ").append(device.getLong("pss_after_kb")).append(" KiB\n")
            .append("Thermal status: ").append(device.getInt("thermal_status_before"))
            .append(" → ").append(device.getInt("thermal_status_after")).append("\n\n")
            .append("Saved: ").append(file.getAbsolutePath());
        return out.toString();
    }

    private void finishSuccess(File file, String summary) {
        lastResultFile = file;
        status.setText("Benchmark complete");
        results.setText(summary);
        runButton.setEnabled(true);
        resetButton.setEnabled(true);
        shareButton.setEnabled(true);
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void finishFailure(Throwable error) {
        status.setText("Benchmark failed");
        results.setText(error.getClass().getSimpleName() + ": " + error.getMessage());
        runButton.setEnabled(true);
        resetButton.setEnabled(true);
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void shareResult() {
        if (lastResultFile == null) return;
        Uri uri = FileProvider.getUriForFile(this, getPackageName() + ".files", lastResultFile);
        Intent send = new Intent(Intent.ACTION_SEND);
        send.setType("application/json");
        send.putExtra(Intent.EXTRA_STREAM, uri);
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(Intent.createChooser(send, "Share ETAR result"));
    }

    private static String utcTimestamp() {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US);
        format.setTimeZone(TimeZone.getTimeZone("UTC"));
        return format.format(new Date());
    }

    private static String fileTimestamp() {
        return new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
    }

    private static String formatBytes(long bytes) {
        return String.format(Locale.US, "%.2f MiB", bytes / 1048576.0);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private native String runNativeBenchmark(
        long corpusSize, int dimension, int queryCount, int warmupQueries,
        int repetitions, int k, int rerank, float rho, int headCap,
        float lambdaTail, float alphaMin, float alphaMax, long seed,
        float measurementSeconds, boolean quantizeQuery,
        int cpuCore,
        String distribution, String methodOrder);

    private static final class Config {
        long corpusSize;
        int dimension;
        int queryCount;
        int warmupQueries;
        int repetitions;
        int k;
        int rerank;
        float rho;
        int headCap;
        float lambdaTail;
        float alphaMin;
        float alphaMax;
        long seed;
        float measurementSeconds;
        boolean quantizeQuery;
        int cpuCore;
        String distribution;
        String methodOrder;
    }

    private static final class BatterySnapshot {
        int capacityPercent = -1;
        int currentNowMicroamps = Integer.MIN_VALUE;
        boolean charging;
    }
}
