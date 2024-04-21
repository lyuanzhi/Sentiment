use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use rust_bert::pipelines::sentiment::SentimentModel;
use serde::Deserialize;
use actix_web_prom::PrometheusMetricsBuilder;
use prometheus::{opts, IntCounterVec};

#[derive(Deserialize)]
struct TextQuery {
    text: String,
}

async fn sentiment_predict(correct_counter: web::Data<IntCounterVec>, query: web::Query<TextQuery>) -> impl Responder {
    let input = query.text.clone();
    let prediction_result = web::block(move || {
        let sentiment_classifier = SentimentModel::new(Default::default()).expect("Error creating model");
        let input = [input.as_str()];
        sentiment_classifier.predict(&input)
    })
    .await;

    match prediction_result {
        Ok(output) => {
            let mut response = String::new();
            for sentiment in output {
                response.push_str(&format!("Sentiment: {:?}\n", sentiment.polarity));
            }
            correct_counter.with_label_values(&["/sentiment"]).inc();
            HttpResponse::Ok().content_type("text/plain").body(response)
        },
        Err(_) => {
            HttpResponse::InternalServerError().content_type("text/plain").body(format!("Operation Error!"))
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let prometheus = PrometheusMetricsBuilder::new("api")
        .endpoint("/metrics")
        .build()
        .unwrap();
    let correct_total_opts = opts!("http_requests_correct_total", "Total number of correct HTTP requests.").namespace("api");
    let correct_total = IntCounterVec::new(correct_total_opts, &["endpoint"]).unwrap();
    prometheus.registry.register(Box::new(correct_total.clone())).unwrap();
    HttpServer::new(move || {
        App::new().wrap(prometheus.clone()).app_data(web::Data::new(correct_total.clone())).service(
            web::resource("/sentiment").to(sentiment_predict)
        )
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[actix_rt::test]
    async fn test1() {
        let app = actix_web::test::init_service(App::new().app_data(web::Data::new(
            IntCounterVec::new(opts!("test", "test"), &["endpoint"]).unwrap(),
        )).service(web::resource("/sentiment").to(sentiment_predict))).await;
        let req = actix_web::test::TestRequest::get()
            .uri("/sentiment?text=I%20am%20sad.")
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        let body = actix_web::test::read_body(resp).await;
        assert_eq!(body, web::Bytes::from_static(b"Sentiment: Negative\n"));
    }

    #[actix_rt::test]
    async fn test2() {
        let app = actix_web::test::init_service(App::new().app_data(web::Data::new(
            IntCounterVec::new(opts!("test", "test"), &["endpoint"]).unwrap(),
        )).service(web::resource("/sentiment").to(sentiment_predict))).await;
        let req = actix_web::test::TestRequest::get()
            .uri("/sentiment?text=")
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        let body = actix_web::test::read_body(resp).await;
        assert_eq!(body, web::Bytes::from_static(b"Sentiment: Positive\n"));
    }

    #[actix_rt::test]
    async fn test3() {
        let app = actix_web::test::init_service(App::new().app_data(web::Data::new(
            IntCounterVec::new(opts!("test", "test"), &["endpoint"]).unwrap(),
        )).service(web::resource("/sentiment").to(sentiment_predict))).await;
        let req = actix_web::test::TestRequest::get()
            .uri("/sentiment")
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        let body = actix_web::test::read_body(resp).await;
        assert_eq!(body, web::Bytes::from_static(b"Query deserialize error: missing field `text`"));
    }
}
