#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[])
{
    if (argc != 3) {
        printf("usage: %s <text.flist> <text>", argv[0]);
        return 1;
    }
    FILE* text_list_file = fopen(argv[1], "r"); /* should check the result */
    FILE* text_file = fopen(argv[2], "w");
    char line[256];
    char content[256];

    while (fgets(line, sizeof(line), text_list_file)) {
        line[strlen(line)-1] = 0;
        FILE* utt_text_file = fopen(line, "r");
        fgets(content, sizeof(content), utt_text_file);
        fputs(content, text_file);
        fclose(utt_text_file);
    }
    fclose(text_list_file);
    fclose(text_file);

    return 0;
}
