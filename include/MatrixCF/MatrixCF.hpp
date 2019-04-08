#pragma once

#include <functional>
#include <omp.h>
#include "EasyCL.hpp"

namespace mcf{
    using namespace ecl;

    enum REDUCE {FULL, COLUMN, ROW};

    template<typename T>
    class Mat{
    private:
        Variable<size_t> h, w, total_size;
        Array<T> array;
        Variable<bool> ref;

        void clearFields();
        std::string getTypeName() const;
        void requireMatrixShape(const Mat<T>&, size_t, size_t, const std::string&, bool is_result = false) const;
        void requireTotalSize(const Mat<T>&, size_t, const std::string&) const;
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

        // methods (extra)
        void reshape(size_t, size_t);
        void ravel(bool is_column = false);

        // methods (mutable)
        void gen(const std::function<T(size_t, size_t)>&);
        void gen(const std::string&, Computer&);

        void zeros();
        void zeros(Computer&);

        void ones();
        void ones(Computer&);

        void eye(const T& value = T(1));
        void eye(const T& value, Computer&);

        // methods (immutable)
        void map(const std::function<T(const T&)>&, Mat<T>&) const;
        void map(const std::string&, Mat<T>&, Computer&) const;

        void transform(const Mat<T>&, const std::function<T(const T&, const T&)>&, Mat<T>&) const;
        void transform(const Mat<T>&, const std::string&, Mat<T>&, Computer&) const;

        void add(const Mat<T>&, Mat<T>&) const;
        void add(const Mat<T>&, Mat<T>&, Computer&) const;

        void sub(const Mat<T>&, Mat<T>&) const;
        void sub(const Mat<T>&, Mat<T>&, Computer&) const;

        void hadamard(const Mat<T>&, Mat<T>&) const;
        void hadamard(const Mat<T>&, Mat<T>&, Computer&) const;

        void reduce(Mat<T>&, REDUCE option = FULL) const;
        void reduce(Mat<T>&, REDUCE option, Computer&) const;

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
std::string mcf::Mat<T>::getTypeName() const{
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
    else throw std::runtime_error("Get matrix typename: computer calculations on matrices with this template aren't supported");
}

template<typename T>
void mcf::Mat<T>::requireMatrixShape(const Mat<T>& X, size_t require_h, size_t require_w, const std::string& where, bool is_result) const{
    size_t r_h = X.getH();
    size_t r_w = X.getW();

    if(r_h != require_h || r_w != require_w){
        std::string what = is_result ? "result matrix" : "matrix";

        std::string e = "Require shape [" + where + "]: ";
        e += "wrong " + what + " shape ";
        e += std::to_string(r_h) + "x" + std::to_string(r_w);
        e += " != ";
        e += std::to_string(require_h) + "x" + std::to_string(require_w);

        throw std::runtime_error(e);
    }
}
template<typename T>
void mcf::Mat<T>::requireTotalSize(const Mat<T>& X, size_t require_total_size, const std::string& where) const{
    size_t r_total_size = X.getTotalSize();

    if(r_total_size != require_total_size){
        std::string e = "Require total size [" + where + "]: ";
        e += "wrong matrix total size ";
        e += std::to_string(r_total_size) + " != " + std::to_string(require_total_size);
        throw std::runtime_error(e);
    }
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
    return array + i * w;
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

// methods (extra)
template<typename T>
void mcf::Mat<T>::reshape(size_t new_h, size_t new_w){
    requireTotalSize(*this, new_h * new_w, "reshape");
    h = new_h;
    w = new_w;
}
template<typename T>
void mcf::Mat<T>::ravel(bool is_column){
    if(is_column) reshape(total_size, 1);
    else reshape(1, total_size);
}

// methods (mutable)
template<typename T>
void mcf::Mat<T>::gen(const std::function<T(size_t, size_t)>& f){
    #pragma omp parallel for collapse(2)
    for(size_t i = 0; h > i; i++){
        for(size_t j = 0; w > j; j++) setE(f(i, j), i, j);
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
    temp += "size_t h = get_global_size(0);\n";
    temp += "size_t w = get_global_size(1);\n";
    temp += "size_t index = i * w + j;\n";
    temp += type + " ret;\n";
    temp += body + "\n";
    temp += "result[index] = ret;";
    temp += "}";

    ecl::Kernel gen = "gen";

    video.compute(temp, gen, {&array}, {h, w});
}

template<typename T>
void mcf::Mat<T>::zeros(){
    gen([](size_t i, size_t j){
        return T(0);
    });
}
template<typename T>
void mcf::Mat<T>::zeros(ecl::Computer& video){
    gen("ret = 0;", video);
}

template<typename T>
void mcf::Mat<T>::ones(){
    gen([](size_t i, size_t j){
        return T(1);
    });
}
template<typename T>
void mcf::Mat<T>::ones(ecl::Computer& video){
    gen("ret = 1;", video);
}

template<typename T>
void mcf::Mat<T>::eye(const T& value){
    gen([&](size_t i, size_t j){
        return i == j ? value : T(0);
    });
}
template<typename T>
void mcf::Mat<T>::eye(const T& value, ecl::Computer& video){
    std::string val = std::to_string(value);
    gen("ret = i == j ? + " + val + " : 0;", video);
}

// methods (immutable)
template<typename T>
void mcf::Mat<T>::map(const std::function<T(const T&)>& f, mcf::Mat<T>& result) const{
    requireMatrixShape(result, h, w, "map", true);

    #pragma omp parallel for
    for(size_t i = 0; total_size > i; i++) result.getArray()[i] = f(getConstArray()[i]);
}
template<typename T>
void mcf::Mat<T>::map(const std::string& body, mcf::Mat<T>& result, ecl::Computer& video) const{
    requireMatrixShape(result, h, w, "map", true);

    std::string type = getTypeName();

    ecl::Program temp = "__kernel void map";
    temp += "(__global " + type + "* a, __global " + type + "* result)";
    temp += "{\n";
    temp += "size_t index = get_global_id(0) * get_global_size(1) + get_global_id(1);\n";
    temp += type + " v = a[index];\n";
    temp += type + " ret;\n";
    temp += body + "\n";
    temp += "result[index] = ret;\n";
    temp += "}";

    ecl::Kernel map = "map";

    video.compute(temp, map, {&array, &result.array}, {h, w});
}

template<typename T>
void mcf::Mat<T>::transform(const Mat<T>& X, const std::function<T(const T&, const T&)>& f, Mat<T>& result) const{
    requireMatrixShape(X, h, w, "transform");
    requireMatrixShape(result, h, w, "transform", true);

    #pragma omp parallel for
    for(size_t i = 0; total_size > i; i++) result.getArray()[i] = f(getConstArray()[i], X.getConstArray()[i]);
}

template<typename T>
void mcf::Mat<T>::transform(const Mat<T>& X, const std::string& body, Mat<T>& result, ecl::Computer& video) const{
    requireMatrixShape(X, h, w, "transform");
    requireMatrixShape(result, h, w, "transform", true);

    std::string type = getTypeName();

    ecl::Program temp = "__kernel void transform";
    temp += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
    temp += "{\n";
    temp += "size_t index = get_global_id(0) * get_global_size(1) + get_global_id(1);\n";
    temp += type + " v1 = a[index];\n";
    temp += type + " v2 = b[index];\n";
    temp += type + " ret;\n";
    temp += body + "\n";
    temp += "result[index] = ret;\n";
    temp += "}";

    ecl::Kernel transform = "transform";

    video.compute(temp, transform, {&array, &X.array, &result.array}, {h, w});
}

template<typename T>
void mcf::Mat<T>::add(const Mat<T>& X, Mat<T>& result) const{
    transform(X, [](const T& v1, const T& v2){
        return v1 + v2;
    }, result);
}
template<typename T>
void mcf::Mat<T>::add(const Mat<T>& X, Mat<T>& result, ecl::Computer& video) const{
    transform(X, "ret = v1 + v2;", result, video);
}

template<typename T>
void mcf::Mat<T>::sub(const Mat<T>& X, Mat<T>& result) const{
    transform(X, [](const T& v1, const T& v2){
        return v1 - v2;
    }, result);
}
template<typename T>
void mcf::Mat<T>::sub(const Mat<T>& X, Mat<T>& result, ecl::Computer& video) const{
    transform(X, "ret = v1 - v2;", result, video);
}

template<typename T>
void mcf::Mat<T>::hadamard(const Mat<T>& X, Mat<T>& result) const{
    transform(X, [](const T& v1, const T& v2){
        return v1 * v2;
    }, result);
}
template<typename T>
void mcf::Mat<T>::hadamard(const Mat<T>& X, Mat<T>& result, ecl::Computer& video) const{
    transform(X, "ret = v1 * v2;", result, video);
}

template<typename T>
void mcf::Mat<T>::reduce(Mat<T>& result, REDUCE option) const{
    if(option == FULL){
        requireMatrixShape(result, 1, 1, "reduce", true);
        result.zeros();

        for(size_t i = 0; total_size > i; i++) result[0][0] += array[i];
    }else if(option == COLUMN){
        requireMatrixShape(result, 1, w, "reduce", true);
        result.zeros();

        #pragma omp parallel for
        for(size_t j = 0; w > j; j++){
            for(size_t i = 0; h > i; i++) result[0][j] += getE(i, j);
        }
    } else if(option == ROW){
        requireMatrixShape(result, h, 1, "reduce", true);
        result.zeros();

        #pragma omp parallel for
        for(size_t i = 0; h > i; i++){
            for(size_t j = 0; w > j; j++) result[i][0] += getE(i, j);
        }
    }
}
template<typename T>
void mcf::Mat<T>::reduce(Mat<T>& result, REDUCE option, ecl::Computer& video) const{
    if(option == FULL){
        throw std::runtime_error("full reduce on computer temporary unavailable");
    }else if(option == COLUMN){
        requireMatrixShape(result, 1, w, "reduce", true);
        result.zeros(video);

        std::string type = getTypeName();

        ecl::Program temp = "__kernel void reduce";
        temp += "(__global " + type + "* a, __global " + type + "* result)";
        temp += "{\n";
        temp += "size_t j = get_global_id(0);\n";
        temp += "size_t w = get_global_size(0);\n";
        temp += type + " sum = 0;\n";
        temp += "for(size_t i = 0; i < " + std::to_string(h) + "; i++){\n";
        temp += "sum += a[i * w + j];\n";
        temp += "}\n";
        temp += "result[j] = sum;\n";
        temp += "}";

        ecl::Kernel reduce = "reduce";

        video.compute(temp, reduce, {&array, &result.array}, {w});

    } else if(option == ROW){
        requireMatrixShape(result, h, 1, "reduce", true);
        result.zeros(video);

        std::string type = getTypeName();

        ecl::Program temp = "__kernel void reduce";
        temp += "(__global " + type + "* a, __global " + type + "* result)";
        temp += "{\n";
        temp += "size_t i = get_global_id(0);";
        temp += "size_t h = get_global_size(0);";
        temp += "size_t w = " + std::to_string(w) + ";";
        temp += "size_t iw = i * w;\n";
        temp += type + " sum = 0;\n";
        temp += "for(size_t j = 0; j < w; j++){\n";
        temp += "sum += a[iw + j];";
        temp += "}\n";
        temp += "result[i] = sum;\n";
        temp += "}";

        ecl::Kernel reduce = "reduce";

        video.compute(temp, reduce, {&array, &result.array}, {h});
    }
}


template<typename T>
mcf::Mat<T>::~Mat(){
    clearFields();
}