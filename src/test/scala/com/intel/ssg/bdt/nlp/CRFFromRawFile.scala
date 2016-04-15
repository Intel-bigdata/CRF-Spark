/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.ssg.bdt.nlp

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object CRFFromRawFile {

  def main(args: Array[String]) {
    val templateFile = "src/test/resources/chunking/template"
    val trainFile = "src/test/resources/chunking/raw/train.data"
    val testFile = "src/test/resources/chunking/raw/test.data"

    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}")
    val sc = new SparkContext(conf)

    val templates: Array[String] = scala.io.Source.fromFile(templateFile).getLines().filter(_.nonEmpty).toArray
    val trainRDD: RDD[Sequence] = sc.textFile(trainFile).filter(_.nonEmpty).map(sentence => {
      val tokens = sentence.split("\t")
      Sequence(tokens.map(token => {
        val tags: Array[String] = token.split('|')
        Token.put(tags.last, tags.dropRight(1))
      }))
    })

    val model: CRFModel = CRF.train(templates, trainRDD, 0.25, 2)

    val testRDDWithoutLabel: RDD[Sequence] = sc.textFile(testFile).filter(_.nonEmpty).map(sentence => {
      val tokens = sentence.split("\t")
      Sequence(tokens.map(token => {
        val tags = token.split('|')
        Token.put(tags.dropRight(1))
      }))
    })

    trainRDD.repartition(1).saveAsTextFile("target/result/train.data")
    testRDDWithoutLabel.saveAsTextFile("target/result/test.data")

    val testRDDwithLabel: RDD[Sequence] = sc.textFile(testFile).filter(_.nonEmpty).map(sentence => {
      val tokens = sentence.split("\t")
      Sequence(tokens.map(token => {
        val tags = token.split('|')
        Token.put(tags.last, tags.dropRight(1))
      }))
    })

    val results = model.predict(testRDDWithoutLabel)
    val score = results
      .zipWithIndex()
      .map(_.swap)
      .join(testRDDwithLabel.zipWithIndex().map(_.swap))
      .map(_._2)
      .map(x => x._1.compare(x._2))
      .reduce(_ + _)
    val total = testRDDWithoutLabel.map(_.toArray.length).reduce(_ + _)
    println(s"Prediction Accuracy: $score / $total")

    sc.stop()
  }
}
