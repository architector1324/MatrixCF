#pragma once

#include <functional>
#include <omp.h>
#include "EasyCL.hpp"

namespace mcf{
    using namespace ecl;
    static Program methods = Program::load("matrix.cl");

    template<typename T>
    class Mat{
    private:
        Variable<size_t> h, w, total_size;
        Array<T> array;
        Variable<bool> ref;

        void clearFields();
        std::string getTypeName();
    public:
        Mat();
        Mat(size_t, size_t);
        Mat(T*, size_t, size_t);

        Mat(const Mat<T>&);
        Mat<T>& operator=(const Mat<T>&);

        Mat(Mat<T>&&);
        Mat<T>& operator=(Mat<T>&&);

        Array<T>& getArray();

        const Array<T>& getConstArray() const;
        const Variable<size_t>& getH() const;
        const Variable<size_t>& getW() const;
        const Variable<size_t>& getTotalSize() const;
        bool isRef() const;

        const T& getE(size_t, size_t) const;
        void setE(const T&, size_t, size_t);

        T* operator[](size_t);
        operator T*();
        operator const T*() const;

        void send(Computer&);
        void receive(Computer&);
        void release(Computer&);
        void grab(Computer&);

        template<typename U>
        friend std::ostream& operator<<(std::ostream&, const Mat<U>&);
        template<typename U>
        friend Computer& operator<<(Computer&, Mat<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Mat<U>&);

        // methods (mutable)
        void gen(const std::function<T(size_t, size_t, const T&)>&);
        void gen(const std::string&, Computer&);

        ~Mat();
    };
}

// IMPLEMENTATION
template<typename T>
void mcf::Mat<T>::clearFields(){
    h = 0;
    w = 0;
    total_size = 0;
    array.clearFields();
    ref = 0;
}

template<typename T>
std::string mcf::Mat<T>::getTypeName(){
    if constexpr (std::is_same<T, bool>::value) return "bool";
    else if constexpr (std::is_same<T, char>::value) return "char";
    else if constexpr (std::is_same<T, unsigned char>::value) return "unsigned char";
    else if constexpr (std::is_same<T, short>::value) return "short";
    else if constexpr (std::is_same<T, unsigned short>::value) return "unsigned short";
    else if constexpr(std::is_same<T, int>::value) return "int";
    else if constexpr(std::is_same<T, unsigned int>::value) return "unsigned int";
    else if constexpr(std::is_same<T, long>::value) return "long";
    else if constexpr(std::is_same<T, unsigned long>::value) return "unsigned long";
    else if constexpr(std::is_same<T, float>::value) return "float";
    else if constexpr (std::is_same<T, double>::value) return "double";
    else if constexpr(std::is_same<T, size_t>::value) return "size_t";
    else throw std::runtime_error("Get matrix type name: computer calculations on matrices with this template aren't supported");

}

// Constructors
template<typename T>
mcf::Mat<T>::Mat(){
    h = 0;
    w = 0;
    total_size = 0;
    ref = false;
}

template<typename T>
mcf::Mat<T>::Mat(size_t h, size_t w) : array(w * h){
    this->h = h;
    this->w = w;
    total_size = w * h;
    ref = false;
}

template<typename T>
mcf::Mat<T>::Mat(T* array, size_t h, size_t w) : array(array, h * w, READ_WRITE){
    this->h = h;
    this->w = w;
    total_size = w * h;
    ref = true;
}


template<typename T>
mcf::Mat<T>::Mat(const Mat<T>& other){
    clearFields();

    h = other.h;
    w = other.w;
    total_size = other.total_size;
    array = other.array;
    ref = false;
}
template<typename T>
mcf::Mat<T>& mcf::Mat<T>::operator=(const Mat<T>& other){
    clearFields();

    h = other.h;
    w = other.w;
    total_size = other.total_size;
    array = other.array;
    ref = false;

    return *this;
}

template<typename T>
mcf::Mat<T>::Mat(Mat<T>&& other){
    h = std::move(other.h);
    w = std::move(other.w);
    total_size = std::move(other.total_size);
    array = std::move(other.array);
    ref = std::move(other.ref);

    other.clearFields();
}

template<typename T>
mcf::Mat<T>& mcf::Mat<T>::operator=(Mat<T>&& other){
    h = std::move(other.h);
    w = std::move(other.w);
    total_size = std::move(other.total_size);
    array = std::move(other.array);
    ref = std::move(other.ref);

    other.clearFields();

    return *this;
}

// Getters
template<typename T>
const mcf::Array<T>& mcf::Mat<T>::getConstArray() const{
    return array;
}
template<typename T>
const mcf::Variable<size_t>& mcf::Mat<T>::getH() const{
    return h;
}
template<typename T>
const mcf::Variable<size_t>& mcf::Mat<T>::getW() const{
    return w;
}
template<typename T>
const mcf::Variable<size_t>& mcf::Mat<T>::getTotalSize() const{
    return total_size;
}

template<typename T>
mcf::Array<T>& mcf::Mat<T>::getArray(){
    return array;
}

template<typename T>
bool mcf::Mat<T>::isRef() const{
    return ref;
}

template<typename T>
const T& mcf::Mat<T>::getE(size_t i, size_t j) const{
    return array[w * i + j];
}
template<typename T>
void mcf::Mat<T>::setE(const T& value, size_t i, size_t j){
    array[w * i + j] = value;
}

template<typename T>
T* mcf::Mat<T>::operator[](size_t i){
    return array + i;
}

template<typename T>
mcf::Mat<T>::operator T*(){
    return array;
}
template<typename T>
mcf::Mat<T>::operator const T*() const{
    return array;
}

template<typename T>
void mcf::Mat<T>::send(Computer& video){
    video << h << w << array;
}
template<typename T>
void mcf::Mat<T>::receive(Computer& video){
    video >> array;
}
template<typename T>
void mcf::Mat<T>::release(Computer& video){
    video.release({&h, &w, &array});
}
template<typename T>
void mcf::Mat<T>::grab(Computer& video){
    video.grab({&h, &w, &array});
}

namespace mcf{
    template<typename T>
    std::ostream& operator<<(std::ostream& s, const Mat<T>& other){
        for(size_t i = 0; other.h > i; i++){
            s << "(" << other.getE(i, 0);
            for(size_t j = 1; other.w > j; j++) s << ", " << other.getE(i, j);
            s << ")\n";
        }
        return s;
    }

    template<typename T>
    Computer& operator<<(Computer& video, Mat<T>& other){
        other.send(video);
        return video;
    }
    template<typename T>
    Computer& operator>>(Computer& video, Mat<T>& other){
        other.receive(video);
        return video;
    }
}

// methods (mutable)
template<typename T>
void mcf::Mat<T>::gen(const std::function<T(size_t, size_t, const T&)>& f){
    #pragma omp parallel for collapse(2)
    for(size_t i = 0; h > i; i++){
        for(size_t j = 0; w > j; j++) setE(f(i, j, getE(i, j)), i, j);
    }
}
template<typename T>
void mcf::Mat<T>::gen(const std::string& body, ecl::Computer& video){
    std::string type = getTypeName();

    ecl::Program temp = "__kernel void gen";
    temp += "(__global " + type + "* result)";
    temp += "{\n";
    temp += "size_t i = get_global_id(0);\n";
    temp += "size_t j = get_global_id(1);\n";
    temp += "size_t w = get_global_size(1);\n";
    temp += "size_t index = i * w + j;\n";
    temp += type + " v = result[index];";
    temp += body + "\n";
    temp += "}";

    ecl::Kernel gen = "gen";

    video.compute(temp, gen, {&array}, {h, w});
}

template<typename T>
mcf::Mat<T>::~Mat(){
    clearFields();
}