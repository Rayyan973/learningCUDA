#include <stdio.h>

int main() {
    int a=10;
    void* voidptr;
    voidptr = &a;
    // this is a void pointer, its not de-reference-able

    //gotta dereference it by first typecasting it into another type of pointer.
    printf("Integer = %d\n", *(int*)voidptr); //first asterisk is dereferencing and the 2nd one (int*) is to cast it into integer pointer type.
    return 0;
}

//the malloc() function actually returns a void pointer. 
//but by default it gets casted into another type (similar to the print statement above)
