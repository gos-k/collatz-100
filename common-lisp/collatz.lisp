; sbcl --load collatz.lisp

(ql:quickload 'lparallel)
(use-package 'lparallel)

(setf *kernel* (make-kernel 4))

(defun rand100 ()
  (random (expt 10 100)))

(defun collatz (value)
  (do ((size 0 (+ size 1))
       (result value (if (= (mod result 2) 1)
                       (+ (* 3 result) 1)
                       (/ result 2))))
    ((<= result 1) size)))

(defun longer (a b)
  (let* ((num 10000)
         (ns (append (loop for i upto num collect (rand100)) (list a b)))
         (cs (pmap 'list #'(lambda (r) (collatz r)) ns))
         (ncs (pmap 'list #'(lambda (r c) (list r c)) ns cs))
         (result (preduce #'(lambda (a b) (if (> (second a) (second b)) a b)) ncs)))
    (format t "~s~%" result)
    (longer (first result) (second result))))

(make-random-state)
(longer 1 1)
