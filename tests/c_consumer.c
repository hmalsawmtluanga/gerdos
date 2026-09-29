/* Pure-C consumer: compiled with cc, not c++. Runs the signal chain
 * through the C ABI and checks the closed-form values. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gerdos/gerdos.h"

static char* load_text(const char* path) {
    FILE* input = fopen(path, "rb");
    if (input == NULL) {
        return NULL;
    }
    fseek(input, 0, SEEK_END);
    long size = ftell(input);
    fseek(input, 0, SEEK_SET);
    char* text = (char*)malloc((size_t)size + 1);
    if (text == NULL) {
        fclose(input);
        return NULL;
    }
    if (fread(text, 1, (size_t)size, input) != (size_t)size) {
        free(text);
        fclose(input);
        return NULL;
    }
    text[size] = 0;
    fclose(input);
    return text;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: c_consumer <workloads-dir>\n");
        return 1;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/signal_chain.gwd", argv[1]);
    char* text = load_text(path);
    if (text == NULL) {
        printf("FAIL: cannot read artifact\n");
        return 1;
    }
    gerdos_runtime* runtime = gerdos_create();
    if (runtime == NULL) {
        printf("FAIL: cannot create runtime\n");
        free(text);
        return 1;
    }
    if (gerdos_load(runtime, text) != 0) {
        printf("FAIL: artifact refused\n");
        gerdos_destroy(runtime);
        free(text);
        return 1;
    }
    free(text);
    if (gerdos_run(runtime) != 3) {
        printf("FAIL: expected 3 coherent completions\n");
        gerdos_destroy(runtime);
        return 1;
    }
    /* Closed forms: 0.75 exact across all three records. */
    unsigned long long data_ids[3] = {901, 913, 921};
    unsigned long long res_ids[3] = {9002, 9013, 9021};
    for (int r = 0; r < 3; ++r) {
        for (int i = 0; i < 6; ++i) {
            float got = gerdos_sample(runtime, data_ids[r], res_ids[r], (size_t)i);
            if (got != 0.75f) {
                printf("FAIL: record %d element %d = %f\n", r, i, got);
                gerdos_destroy(runtime);
                return 1;
            }
        }
    }
    if (gerdos_evidence(runtime) != 3) {
        printf("FAIL: evidence != 3\n");
        gerdos_destroy(runtime);
        return 1;
    }
    /* Fail-closed: garbage refuses with a line number. */
    if (gerdos_load(runtime, "bogus 1 2\n") == 0) {
        printf("FAIL: garbage accepted\n");
        gerdos_destroy(runtime);
        return 1;
    }
    /* Host identity: the machine the runtime actually saw. */
    const char* host = gerdos_host_text(runtime);
    if (host == NULL || host[0] == '\0') {
        printf("FAIL: no host identity\n");
        gerdos_destroy(runtime);
        return 1;
    }
    printf("host: %s\n", host);
    if (strcmp(gerdos_host_text(NULL), "(no runtime)") != 0) {
        printf("FAIL: null host text not handled\n");
        gerdos_destroy(runtime);
        return 1;
    }
    if (gerdos_sample(NULL, 0, 0, 0) != 0.0f || gerdos_evidence(NULL) != 0 || gerdos_run(NULL) != -1) {
        printf("FAIL: null runtime not handled\n");
        gerdos_destroy(runtime);
        return 1;
    }
    printf("c-consumer: 0.75 exact, 3 coherent, evidence 3, refusals loud\n");
    gerdos_destroy(runtime);
    return 0;
}
