#include <stdio.h>

int main() {
    int a=10;
    void* voidptr;
    voidptr = &a;
    // this is a void pointer, its not de-reference-able (which is useful because it literally points to no-type
    // so you can assign it the address of smth and then cast the voidptr into that type and dereference it that way)

    //gotta dereference it by first typecasting it into another type of pointer.
    printf("Integer = %d\n", *(int*)voidptr); //first asterisk is dereferencing and the 2nd one (int*) is to cast it into integer *pointer* type.
    return 0;
}

//the malloc() function actually returns a void pointer. 
//but by default it gets casted into another type (similar to the print statement above)
