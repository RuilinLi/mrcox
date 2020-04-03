#include <Rcpp.h>
#include <RcppEigen.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <cmath>
#include "mrcox_types.h"
// [[Rcpp::depends(RcppEigen)]]

void _cumsum(VectorXd & x){
    double current = 0;
    int n = x.size();
    for(int i = 0; i < n; ++i){
        current += x(i);
        x(i) = current;
    }
}

void _rev_cumsum_assign(const VectorXd &src, VectorXd &dest){
    double current = 0;
    int n = src.size();
    for (int i = 0; i < n; ++i){
        current += src(n-1-i);
        dest(n-1-i) = current;
    }
}

class Cox_problem{
    // Put the input variables in these  containers
    std::vector<MapMatd> X_all;
    std::vector<VectorXd> censor_all; // Want to modify this
    std::vector<MapVeci> rankmin_all;
    std::vector<MapVeci> rankmax_all;
    // Store intermediate results
    std::vector<VectorXd> outer_accumu_all;
    std::vector<VectorXd> eta_all;
    std::vector<VectorXd> exp_eta_all;
    std::vector<VectorXd> exp_accumu_all;
    std::vector<VectorXd> residual_all;

    const int K;  // Number of response
    std::vector<int> ncase_all;

    // Note all these functions assume the ordering is 1,2,3,... presorted sequence
    void _update_exp(int k)
    {
        int ncase = ncase_all[k];
        _rev_cumsum_assign(exp_eta_all[k], exp_accumu_all[k]);

        // Don't need this if there is no tied events
        for (int i = 0; i < ncase; ++i){
            exp_accumu_all[k](i) = exp_accumu_all[k](rankmin_all[k](i));
        }
    }

    void _update_outer(int k)
    {
        outer_accumu_all[k].noalias() = (censor_all[k].array() / exp_accumu_all[k].array()).matrix();
        _cumsum(outer_accumu_all[k]);

        int ncase = ncase_all[k];
        // Don't need this if there is no tied events
        for (int i = 0; i < ncase; ++i){
            outer_accumu_all[k](i) = outer_accumu_all[k](rankmax_all[k](i));
        }
    }

    double get_residual(const MatrixXd & v, bool get_val = false)
    {
        #pragma omp parallel for
        for (int k = 0; k < K; ++k){
            eta_all[k].noalias() = X_all[k] * v.row(k).transpose();
            exp_eta_all[k].noalias() = eta_all[k].array().exp().matrix();
            _update_exp(k);
            _update_outer(k);
            residual_all[k].noalias() = (exp_eta_all[k].array()*outer_accumu_all[k].array() - censor_all[k].array()).matrix();
        }

        double cox_val = 0.0;
        if(get_val){
            #pragma omp parallel for reduction(+:cox_val)
            for (int k = 0; k < K; ++k){
                cox_val += ((exp_accumu_all[k].array().log() - eta_all[k].array()) * censor_all[k].array()).sum();
            }
        }
        return cox_val;
    }

    public:
    Cox_problem(const Rcpp::List & X_list,
                const Rcpp::List & censoring_list,
                const Rcpp::List & rankmin_list,
                const Rcpp::List & rankmax_list) : K(X_list.size())
    {
        X_all.reserve(K);
        censor_all.reserve(K);
        rankmin_all.reserve(K);
        rankmax_all.reserve(K);
        outer_accumu_all.resize(K);
        eta_all.resize(K);
        exp_eta_all.resize(K);
        exp_accumu_all.resize(K);
        residual_all.resize(K);
        ncase_all.resize(K);
        int ncase;
        for (int k = 0; k < K; ++k){
            X_all.push_back(Rcpp::as<MapMatd>(X_list[k]));
            censor_all.push_back(Rcpp::as<VectorXd>(censoring_list[k]));
            rankmin_all.push_back(Rcpp::as<MapVeci>(rankmin_list[k]));
            rankmax_all.push_back(Rcpp::as<MapVeci>(rankmax_list[k]));
            ncase = rankmin_all[k].size();
            ncase_all[k] = ncase;
            censor_all[k] /= (double)ncase; // Do this to adjust gradient with the number of observations
            outer_accumu_all[k].resize(ncase);
            eta_all[k].resize(ncase);
            exp_eta_all[k].resize(ncase);
            exp_accumu_all[k].resize(ncase);
            residual_all[k].resize(ncase);
        }
    }

    std::vector<VectorXd> Rget_residual(const MatrixXd & v){
        get_residual(v);
        return residual_all;
    }

    double get_gradient(const MatrixXd & v, MatrixXd & grad, bool get_val = false)
    {
        double result = get_residual(v, get_val);
        #pragma omp parallel for
        for (int k=0; k < K; ++k){
            grad.row(k).noalias() = (residual_all[k].transpose() * X_all[k]);
        }
        return result;
    }

    double get_value_only(const MatrixXd &v)
    {
        double cox_val = 0.0;
        #pragma omp parallel for
        for (int k = 0; k < K; ++k){
            eta_all[k].noalias() = X_all[k] * v.row(k).transpose();
            exp_eta_all[k].noalias() = eta_all[k].array().exp().matrix();
            _update_exp(k);
        }
        #pragma omp parallel for reduction(+:cox_val)
        for (int k = 0; k < K; ++k){
            cox_val += ((exp_accumu_all[k].array().log() - eta_all[k].array()) * censor_all[k].array()).sum();
        }
        return cox_val;
    }
};

void update_parameters(MatrixXd & B, const MatrixXd & grad, const MatrixXd &v, const double step_size,
                       double lambda_1, double lambda_2, const Eigen::RowVectorXd & penalty_factor,
                       VectorXd & B_col_norm)
{
    B.noalias() = v - step_size*grad;
    // Apply proximal operator here:
    //Soft-thresholding
    B = ((B.cwiseAbs().rowwise() - lambda_1*step_size*penalty_factor).array().max(0) * B.array().sign()).matrix();
    // Group soft-thresholding
    // should be called the pmax of B_col_norm  and lambda_2*step_size
    B_col_norm.noalias() = B.colwise().norm().cwiseMax(lambda_2*step_size*penalty_factor);
    B = B * ((B_col_norm.array() - lambda_2*step_size*penalty_factor.array())/(B_col_norm.array())).matrix().asDiagonal();
}

// [[Rcpp::export]]
Rcpp::List test(const Rcpp::List & X_list,
               const Rcpp::List & censoring_list,
               MatrixXd B,
               const Rcpp::List & rankmin_list,
               const Rcpp::List & rankmax_list,
               double step_size,
               VectorXd lambda_1_all,
               VectorXd lambda_2_all,
               Eigen::RowVectorXd penalty_factor, // Penalty factor for each group of variables
               int niter, // Maximum number of iterations
               double linesearch_beta,
               double eps, // convergence criteria
               double tol = 1e-10// line search tolerance
               )
{
    Eigen::initParallel();
    const int K = B.rows();
    const int p = B.cols();
    Cox_problem prob(X_list, censoring_list, rankmin_list, rankmax_list);

    MatrixXd grad(K, p);
    MatrixXd grad_ls(K, p); // For line search
    VectorXd B_col_norm(p);
    MatrixXd previous_B(B);
    MatrixXd v(B); //This matrix is used to hold temporary matrix from acceleration
    double cox_val;
    double cox_val_next;
    double rhs_ls; // right-hand side of line search condition
    double lambda_1;
    double lambda_2;
    const int num_lambda = lambda_1_all.size();
    double step_size_intial = step_size;
    Rcpp::List result(num_lambda);
    bool stop; // Stop line searching
    double weight_old, weight_new;
    double obj, prev_obj;
    // Initialization done, starting solving the path
    struct timeval start, end;
    for (int lam_ind = 0; lam_ind < num_lambda; ++lam_ind){
        gettimeofday(&start, NULL);

        lambda_1 = lambda_1_all[lam_ind];
        lambda_2 = lambda_2_all[lam_ind];
        prev_obj = prob.get_value_only(B);
        weight_old = 1.0;
        step_size = step_size_intial;

        for (int i = 0; i< niter; i++){
            previous_B.noalias() = B;
            cox_val = prob.get_gradient(v, grad, true);
            while (true){
                update_parameters(B, grad, v, step_size, lambda_1, lambda_2, penalty_factor, B_col_norm);

                cox_val_next = prob.get_value_only(B);
                stop = false;
                if(abs((cox_val_next - cox_val)/fmax(1.0, abs(cox_val_next))) > tol){
                    rhs_ls = cox_val + (grad.array() * (B - v).array()).sum() + (B-v).squaredNorm()/(2*step_size);
                    stop = (cox_val_next <= rhs_ls);
                } else {
                    prob.get_gradient(B, grad_ls, false);
                    rhs_ls = ((B-v).array() * (grad_ls - grad).array()).sum();
                    stop = (abs(rhs_ls) <= (B-v).squaredNorm()/(2*step_size));
                }

                if (stop){
                    obj = cox_val_next + \
                    (B.colwise().norm().array()*penalty_factor.array()).sum()*lambda_2 + \
                    (B.cwiseAbs().colwise().sum().array()*penalty_factor.array()).sum()*lambda_1;
                    break;
                }
                step_size /= linesearch_beta;
            }

            if(abs((prev_obj - obj)/fmax(abs(prev_obj), 1.0)) < eps){
                std::cout << "convergence based on value change reached in " << i <<" iterations\n";
                std::cout << "current step size is " << step_size << std::endl;
                gettimeofday(&end, NULL);
                double delta  = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout <<  "elapsed time is " << delta << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
                break;
            }
            prev_obj = obj;

            // Nesterov weight
            weight_new = 0.5*(1+sqrt(1+4*weight_old*weight_old));
            v.noalias() = B + ((weight_old - 1)/weight_new) * (B - previous_B);
            weight_old = weight_new;
            // v.noalias() = B + ((double)i/(double)(i+3))*(B - previous_B); // Another strategy
            if (i != 0 && i % 100 == 0){
                std::cout << "reached " << i << " iterations\n";
                gettimeofday(&end, NULL);
                double delta  = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout <<  "elapsed time is " << delta  << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
            }

        }
        result[lam_ind] = B;
        std::cout << "Solution for the " <<  lam_ind+1 << "th lambda pair is obtained\n";
    }
    return result;
}



// [[Rcpp::export]]
MatrixXd compute_gradient(const Rcpp::List & X_list,
                const Rcpp::List & censoring_list,
                const Rcpp::List & rankmin_list,
                const Rcpp::List & rankmax_list,
                MatrixXd B)
{
    Cox_problem prob(X_list, censoring_list, rankmin_list, rankmax_list);
    int K = B.rows();
    int p = B.cols();
    MatrixXd result(K, p);
    prob.get_gradient(B, result);
    return result;
}

// the norm is defined as \|x\|_1 + alpha \|x\|_2
// This can be used for both KKT check and the strong rule
Eigen::Array<bool, 1, Eigen::Dynamic> is_dual_norm_less_than(MatrixXd grad,
                                                      double alpha,
                                                      double bound,
                                                      Eigen::Map<Eigen::RowVectorXd> penalty_factor,
                                                      double tol)
{
    Eigen::Array<bool, 1, Eigen::Dynamic> result(grad.cols());
    grad = ((grad.cwiseAbs().rowwise() - bound*penalty_factor).array().max(0)).matrix();
    result = grad.colwise().norm().array() <= (alpha*bound*penalty_factor.array() + tol);
    return result;
}

// [[Rcpp::export]]
Rcpp::List compute_residual(const Rcpp::List & X_list,
                const Rcpp::List & censoring_list,
                const Rcpp::List & rankmin_list,
                const Rcpp::List & rankmax_list,
                MatrixXd B)
{
    Cox_problem prob(X_list, censoring_list, rankmin_list, rankmax_list);
    return Rcpp::wrap(prob.Rget_residual(B));
}

// [[Rcpp::export]]
VectorXd compute_dual_norm(const MapMatd grad,
                           double alpha,
                           double tol)
{
    int p = grad.cols();
    VectorXd upperbound((grad.cwiseAbs().colwise().maxCoeff()).cwiseMin(grad.colwise().norm()/alpha));
    VectorXd dual_norm(p);
    for (int i = 0; i < p; ++i){
        double lower = 0.0;
        double upper = upperbound[i];
        if (upper <= tol){
            dual_norm[i] = 0.0;
        } else {
            int num_iter = (int)ceil(log2(upper/tol));
            for (int j = 0; j < num_iter; ++j){
                double bound = (lower + upper)/2;
                bool less = ((grad.col(i).array().abs() - bound).max(0).matrix().norm()) <= alpha * bound;
                if (less){
                    upper = bound;
                } else {
                    lower = bound;
                }
            }
            dual_norm[i] = (lower + upper)/2;
        }
    }
    return dual_norm;
}

